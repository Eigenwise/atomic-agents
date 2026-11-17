import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
import requests
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NullObject, NumberObject, TextStringObject
from reportlab.pdfgen import canvas

from tool.pdf_reader import PdfReaderTool, PdfReaderToolInputSchema


FIRST = "https://example.invalid/research?version=2&section=methods"
SECOND = "https://example.invalid/data#table-1"


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    original = requests.Session.request

    def guarded(self, method, url, **kwargs):
        assert urlsplit(url).hostname == "127.0.0.1", "Embedded links must not be fetched"
        return original(self, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", guarded)


@pytest.fixture
def linked_pdf(tmp_path):
    data = io.BytesIO()
    document = canvas.Canvas(data)
    document.setTitle("Linked research")
    document.drawString(72, 720, "Read the paper or contact its author")
    document.linkURL(FIRST, (72, 700, 150, 720), relative=0)
    document.linkURL("mailto:author@example.invalid", (160, 700, 280, 720), relative=0)
    document.linkURL(FIRST, (72, 670, 150, 690), relative=0)
    document.linkRect("See appendix", "appendix", (72, 640, 150, 660), relative=0)
    document.showPage()
    document.bookmarkPage("appendix")
    document.drawString(72, 720, "Download the data")
    document.linkURL(SECOND, (72, 700, 150, 720), relative=0)
    document.showPage()
    document.drawString(72, 720, "Plain conclusion")
    document.save()
    path = tmp_path / "linked.pdf"
    path.write_bytes(data.getvalue())
    return path


@pytest.fixture(params=["file", "http"])
def source(linked_pdf, request):
    if request.param == "file":
        yield str(linked_pdf)
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = linked_pdf.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/linked.pdf"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_extracts_annotation_targets_missing_from_visible_text(source):
    result = PdfReaderTool().run(PdfReaderToolInputSchema(source=source, include_links=True))

    assert result.error is None
    assert result.total_page_count == result.page_count == 3
    assert [page.links for page in result.pages] == [[FIRST, "mailto:author@example.invalid", FIRST], [SECOND], []]
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.metadata.title == "Linked research"
    assert "Read the paper" in result.text
    assert "Download the data" in result.text
    assert FIRST not in result.text
    assert SECOND not in result.text


def test_links_respect_page_range_and_independent_metadata_choice(source):
    result = PdfReaderTool().run(
        PdfReaderToolInputSchema(source=source, include_links=True, page_range="2", include_metadata=False)
    )

    assert result.error is None
    assert result.page_count == 1
    assert result.pages[0].page_number == 2
    assert result.pages[0].links == [SECOND]
    assert result.metadata is None


@pytest.mark.parametrize("options", [{}, {"include_links": False}])
def test_link_extraction_is_opt_in_and_keeps_original_text(linked_pdf, options):
    result = PdfReaderTool().run(PdfReaderToolInputSchema(source=str(linked_pdf), **options))
    reader = PdfReader(linked_pdf)

    assert result.error is None
    assert [page.links for page in result.pages] == [[], [], []]
    assert [page.text for page in result.pages] == [page.extract_text() for page in reader.pages]


@pytest.mark.parametrize("annotations", [NullObject(), TextStringObject("invalid"), DictionaryObject()])
def test_invalid_annotation_container_does_not_discard_page_text(linked_pdf, annotations):
    writer = PdfWriter(clone_from=linked_pdf)
    writer.pages[0][NameObject("/Annots")] = annotations
    writer.write(linked_pdf)

    result = PdfReaderTool().run(PdfReaderToolInputSchema(source=str(linked_pdf), include_links=True))

    assert result.error is None
    assert result.pages[0].links == []
    assert "Read the paper" in result.pages[0].text
    assert result.pages[1].links == [SECOND]


def test_indirect_actions_and_malformed_entries_keep_other_link_targets(linked_pdf):
    writer = PdfWriter(clone_from=linked_pdf)
    page = writer.pages[0]
    annotations = page["/Annots"]
    action = DictionaryObject({NameObject("/S"): NameObject("/URI"), NameObject("/URI"): TextStringObject("../relative")})
    annotations.extend(
        [
            NullObject(),
            NumberObject(7),
            TextStringObject("not an annotation"),
            DictionaryObject({NameObject("/Subtype"): NameObject("/Link"), NameObject("/A"): NullObject()}),
            DictionaryObject({NameObject("/Subtype"): NameObject("/Text"), NameObject("/A"): action}),
            DictionaryObject({NameObject("/Subtype"): NameObject("/Link"), NameObject("/A"): writer._add_object(action)}),
        ]
    )
    # Both the annotation array and action can be indirect PDF objects.
    page[NameObject("/Annots")] = writer._add_object(ArrayObject(annotations))
    writer.write(linked_pdf)

    result = PdfReaderTool().run(PdfReaderToolInputSchema(source=str(linked_pdf), include_links=True))

    assert result.error is None
    assert result.pages[0].links == [FIRST, "mailto:author@example.invalid", FIRST, "../relative"]
    assert result.pages[1].links == [SECOND]
