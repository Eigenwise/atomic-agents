"""Exercise layout extraction with real text positioned in a PDF."""

import io
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from tool.pdf_reader import PdfReaderTool, PdfReaderToolInputSchema


@pytest.fixture
def table_pdf(tmp_path):
    data = io.BytesIO()
    document = canvas.Canvas(data)
    document.setTitle("Inventory")
    for suffix in ("one", "two"):
        # Write each column separately, as many report generators do. The rendered
        # row associations are different from the content-stream operator order.
        for index, value in enumerate(["Item", "Oranges " + suffix, "Apples " + suffix]):
            document.drawString(72, 720 - 20 * index, value)
        for index, value in enumerate(["Amount", "12", "42"]):
            document.drawString(300, 720 - 20 * index, value)
        document.showPage()
    document.showPage()  # A blank page is a valid empty extraction.
    document.save()
    path = tmp_path / "inventory.pdf"
    path.write_bytes(data.getvalue())
    return path


@pytest.fixture(params=["local", "http"])
def pdf_source(table_pdf, request):
    if request.param == "local":
        yield str(table_pdf)
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = table_pdf.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/inventory.pdf"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_layout_keeps_table_values_on_their_rendered_rows(pdf_source):
    output = PdfReaderTool().run(PdfReaderToolInputSchema(source=pdf_source, extraction_mode="layout"))

    assert output.error is None
    assert output.total_page_count == 3
    assert output.page_count == 3
    assert output.metadata.title == "Inventory"
    for page, suffix in zip(output.pages, ("one", "two")):
        assert re.search(r"^Item +Amount$", page.text, re.MULTILINE)
        assert re.search(rf"^Oranges {suffix} +12$", page.text, re.MULTILINE)
        assert re.search(rf"^Apples {suffix} +42$", page.text, re.MULTILINE)
        value_positions = [line.index(value) for line, value in zip(page.text.splitlines(), ["Amount", "12", "42"])]
        assert len(set(value_positions)) == 1
    assert output.pages[2].text == ""
    assert output.text == "\n\f\n".join(page.text for page in output.pages)


def test_layout_respects_page_selection_and_metadata_option(pdf_source):
    output = PdfReaderTool().run(
        PdfReaderToolInputSchema(source=pdf_source, page_range="2", include_metadata=False, extraction_mode="layout")
    )

    assert output.error is None
    assert output.page_count == 1
    assert output.pages[0].page_number == 2
    assert output.metadata is None
    assert "one" not in output.text
    assert re.search(r"^Oranges two +12$", output.text, re.MULTILINE)


@pytest.mark.parametrize("kwargs", [{}, {"extraction_mode": "plain"}])
def test_default_and_explicit_plain_modes_preserve_existing_output(table_pdf, kwargs):
    output = PdfReaderTool().run(PdfReaderToolInputSchema(source=str(table_pdf), **kwargs))
    reader = PdfReader(table_pdf)

    assert output.error is None
    assert [page.text for page in output.pages] == [page.extract_text() or "" for page in reader.pages]


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_invalid_extraction_modes_are_rejected(mode):
    with pytest.raises(ValidationError, match="extraction_mode"):
        PdfReaderToolInputSchema(source="inventory.pdf", extraction_mode=mode)


def test_tool_schema_advertises_both_extraction_modes():
    field = PdfReaderToolInputSchema.model_json_schema()["properties"]["extraction_mode"]

    assert field["enum"] == ["plain", "layout"]
    assert field["default"] == "plain"
