import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import requests

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from tool.pdf_reader import (
    PdfReaderTool,
    PdfReaderToolConfig,
    PdfReaderToolInputSchema,
    PdfReaderToolOutputSchema,
)


def _make_pdf(pages: list[str], title: str = "Test Doc", author: str = "tester") -> bytes:
    """Build a small in-memory PDF with the given text on each page."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setTitle(title)
    c.setAuthor(author)
    for index, body in enumerate(pages):
        c.drawString(72, 720, f"Page {index + 1}")
        for offset, line in enumerate(body.splitlines() or [body]):
            c.drawString(72, 700 - 14 * offset, line)
        c.showPage()
    c.save()
    return buffer.getvalue()


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(_make_pdf(["Hello World", "Second Page Content", "Third Page"]))
    return path


@pytest.fixture
def tool():
    return PdfReaderTool(config=PdfReaderToolConfig())


def test_parse_page_range_singletons(tool):
    assert PdfReaderTool.parse_page_range("3", 10) == [3]


def test_parse_page_range_range(tool):
    assert PdfReaderTool.parse_page_range("2-4", 10) == [2, 3, 4]


def test_parse_page_range_mixed(tool):
    assert PdfReaderTool.parse_page_range("1,3,5-7", 10) == [1, 3, 5, 6, 7]


def test_parse_page_range_clips_to_total(tool):
    assert PdfReaderTool.parse_page_range("1,3,5-7", 4) == [1, 3]


def test_parse_page_range_rejects_empty(tool):
    with pytest.raises(ValueError):
        PdfReaderTool.parse_page_range("", 10)


def test_parse_page_range_rejects_invalid(tool):
    with pytest.raises(ValueError):
        PdfReaderTool.parse_page_range("foo", 10)
    with pytest.raises(ValueError):
        PdfReaderTool.parse_page_range("5-2", 10)
    with pytest.raises(ValueError):
        PdfReaderTool.parse_page_range("0", 10)


def test_parse_page_range_no_pages_selected(tool):
    """Range entirely outside the doc should fail loudly so the agent knows."""
    with pytest.raises(ValueError):
        PdfReaderTool.parse_page_range("8-9", 4)


def test_run_local_file_reads_all_pages(tool, sample_pdf):
    out = tool.run(PdfReaderToolInputSchema(source=str(sample_pdf)))
    assert isinstance(out, PdfReaderToolOutputSchema)
    assert out.error is None
    assert out.total_page_count == 3
    assert out.page_count == 3
    assert "Hello World" in out.pages[0].text
    assert "Second Page Content" in out.pages[1].text
    assert "Third Page" in out.pages[2].text
    assert "Hello World" in out.text and "\f" in out.text


def test_run_local_file_with_page_range(tool, sample_pdf):
    out = tool.run(PdfReaderToolInputSchema(source=str(sample_pdf), page_range="2"))
    assert out.error is None
    assert out.page_count == 1
    assert out.total_page_count == 3
    assert "Second Page Content" in out.pages[0].text
    # First page text should NOT appear when only page 2 was requested
    assert "Hello World" not in out.text


def test_run_metadata_present(tool, sample_pdf):
    out = tool.run(PdfReaderToolInputSchema(source=str(sample_pdf)))
    assert out.metadata is not None
    assert out.metadata.title == "Test Doc"
    assert out.metadata.author == "tester"


def test_run_metadata_can_be_disabled(tool, sample_pdf):
    out = tool.run(PdfReaderToolInputSchema(source=str(sample_pdf), include_metadata=False))
    assert out.metadata is None


def test_run_missing_local_file_returns_error(tool, tmp_path):
    out = tool.run(PdfReaderToolInputSchema(source=str(tmp_path / "missing.pdf")))
    assert out.error is not None
    assert "not found" in out.error.lower()


def test_run_invalid_page_range_returns_error(tool, sample_pdf):
    out = tool.run(PdfReaderToolInputSchema(source=str(sample_pdf), page_range="abc"))
    assert out.error is not None


def test_run_url_source(monkeypatch, tool, sample_pdf):
    pdf_bytes = sample_pdf.read_bytes()

    class FakeResp:
        headers = {"Content-Length": str(len(pdf_bytes))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield pdf_bytes

    def fake_get(url, headers, timeout, stream):
        assert url.startswith("https://example.com/")
        return FakeResp()

    import tool.pdf_reader as pdf_module

    monkeypatch.setattr(pdf_module.requests, "get", fake_get)
    out = tool.run(PdfReaderToolInputSchema(source="https://example.com/sample.pdf"))
    assert out.error is None
    assert out.total_page_count == 3


def test_run_url_too_large_by_header(monkeypatch, tool):
    class FakeResp:
        headers = {"Content-Length": str(1_000_000_000)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            return iter(())

    monkeypatch.setattr("tool.pdf_reader.requests.get", lambda *a, **kw: FakeResp())
    out = tool.run(PdfReaderToolInputSchema(source="https://example.com/huge.pdf"))
    assert out.error is not None
    assert "too large" in out.error.lower()


@pytest.mark.parametrize("failure", ["status", "declared_size", "streamed_size", "invalid_length", None])
def test_url_responses_close_on_success_and_download_failures(monkeypatch, sample_pdf, failure):
    pdf_bytes = sample_pdf.read_bytes()
    # A partial read must leave unread data, rather than closing at EOF by coincidence.
    payload = b"x" * (3 * 64 * 1024) if failure == "streamed_size" else pdf_bytes

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404 if failure == "status" else 200)
            if failure == "declared_size":
                self.send_header("Content-Length", "1000000000")
            elif failure == "invalid_length":
                self.send_header("Content-Length", "invalid")
            elif failure != "streamed_size":
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    responses = []
    session = requests.Session()
    session.trust_env = False

    def get_response(*args, **kwargs):
        response = session.get(*args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr("tool.pdf_reader.requests.get", get_response)
    reader = PdfReaderTool(PdfReaderToolConfig(max_size_bytes=1 if failure == "streamed_size" else 1024 * 1024))
    try:
        out = reader.run(PdfReaderToolInputSchema(source=f"http://127.0.0.1:{server.server_port}/sample.pdf"))
        assert len(responses) == 1
        if failure is None:
            assert out.error is None
            assert out.page_count == 3
            assert "Hello World" in out.text
        else:
            assert out.error is not None
        assert responses[0].raw.closed
    finally:
        for response in responses:
            response.close()
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
