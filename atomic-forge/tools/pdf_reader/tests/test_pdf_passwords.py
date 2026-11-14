import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import SecretStr
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from tool.pdf_reader import PdfReaderTool, PdfReaderToolConfig, PdfReaderToolInputSchema


USER_PASSWORD = "reader-test-password"
OWNER_PASSWORD = "owner-test-password"


def make_pdf(algorithm=None, user_password=USER_PASSWORD):
    buffer = io.BytesIO()
    document = canvas.Canvas(buffer)
    document.setTitle("Protected report")
    document.setAuthor("PDF reader fixture")
    for text in ("First page content", "Second page content"):
        document.drawString(72, 720, text)
        document.showPage()
    document.save()
    if algorithm is None:
        return buffer.getvalue()
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(buffer.getvalue())))
    writer.encrypt(user_password, owner_password=OWNER_PASSWORD, algorithm=algorithm)
    encrypted = io.BytesIO()
    writer.write(encrypted)
    return encrypted.getvalue()


@pytest.fixture
def source_factory(tmp_path):
    servers = []
    requests = []

    def source(data, kind):
        if kind == "local":
            path = tmp_path / "report.pdf"
            path.write_bytes(data)
            return str(path)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, dict(self.headers)))
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_port}/report.pdf"

    yield source, requests
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("kind", ["local", "http"])
@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-256"])
@pytest.mark.parametrize("password", [USER_PASSWORD, OWNER_PASSWORD])
def test_password_reads_selected_text_and_metadata(source_factory, kind, algorithm, password):
    if algorithm.startswith("AES"):
        pytest.importorskip("cryptography")
    source, requests = source_factory
    location = source(make_pdf(algorithm), kind)
    tool = PdfReaderTool(PdfReaderToolConfig(password=SecretStr(password)))

    result = tool.run(PdfReaderToolInputSchema(source=location, page_range="2"))

    assert result.error is None
    assert result.source == location
    assert result.total_page_count == 2
    assert result.page_count == 1
    assert [page.page_number for page in result.pages] == [2]
    assert result.text == result.pages[0].text
    assert "Second page content" in result.text
    assert "First page content" not in result.text
    assert result.metadata.title == "Protected report"
    assert result.metadata.author == "PDF reader fixture"
    assert password not in result.model_dump_json()
    if kind == "http":
        assert len(requests) == 1
        assert password not in repr(requests)
        assert "Authorization" not in requests[0][1]


@pytest.mark.parametrize("kind", ["local", "http"])
@pytest.mark.parametrize("password", [None, "incorrect-test-password", ""])
def test_missing_or_wrong_password_returns_empty_error_result(source_factory, kind, password):
    source, _ = source_factory
    tool = PdfReaderTool(PdfReaderToolConfig(password=password))
    location = source(make_pdf("RC4-128"), kind)

    result = tool.run(PdfReaderToolInputSchema(source=location))

    assert result.error
    if password is not None:
        assert "Incorrect password" in result.error
    assert result.source == location
    assert result.text == ""
    assert result.pages == []
    assert result.page_count == result.total_page_count == 0
    assert result.metadata is None
    for secret in (USER_PASSWORD, OWNER_PASSWORD, "incorrect-test-password"):
        assert secret not in result.model_dump_json()


@pytest.mark.parametrize("password", [None, USER_PASSWORD, ""])
def test_unencrypted_pdf_does_not_require_matching_password(source_factory, password):
    source, _ = source_factory
    tool = PdfReaderTool(PdfReaderToolConfig(password=password))
    result = tool.run(PdfReaderToolInputSchema(source=source(make_pdf(), "local")))
    assert result.error is None
    assert result.page_count == result.total_page_count == 2
    assert "First page content" in result.text
    assert "Second page content" in result.text
    assert "\f" in result.text


@pytest.mark.parametrize("password", [None, ""])
def test_empty_user_password_remains_readable(source_factory, password):
    source, _ = source_factory
    location = source(make_pdf("RC4-128", user_password=""), "local")
    result = PdfReaderTool(PdfReaderToolConfig(password=password)).run(PdfReaderToolInputSchema(source=location))
    assert result.error is None
    assert result.page_count == 2
    assert "First page content" in result.text


def test_password_config_masks_value_and_input_schema_does_not_expose_it():
    config = PdfReaderToolConfig(password=USER_PASSWORD)
    assert isinstance(config.password, SecretStr)
    assert config.password.get_secret_value() == USER_PASSWORD
    assert USER_PASSWORD not in repr(config)
    assert USER_PASSWORD not in config.model_dump_json()
    schema = PdfReaderTool(config).input_schema.model_json_schema()
    assert "password" not in schema["properties"]


def test_password_works_without_metadata(source_factory):
    source, _ = source_factory
    result = PdfReaderTool(PdfReaderToolConfig(password=USER_PASSWORD)).run(
        PdfReaderToolInputSchema(source=source(make_pdf("RC4-128"), "local"), include_metadata=False)
    )
    assert result.error is None
    assert result.metadata is None
    assert result.page_count == 2
    assert "Second page content" in result.text


def test_separate_tools_keep_separate_passwords(source_factory):
    source, _ = source_factory
    location = source(make_pdf("RC4-128"), "local")
    valid = PdfReaderTool(PdfReaderToolConfig(password=USER_PASSWORD))
    invalid = PdfReaderTool(PdfReaderToolConfig(password="another-test-password"))
    params = PdfReaderToolInputSchema(source=location)
    assert valid.run(params).error is None
    assert invalid.run(params).error
    assert valid.run(params).error is None


def test_unicode_password_uses_pypdf_decryption(source_factory):
    pytest.importorskip("cryptography")
    source, _ = source_factory
    password = "阅读报告-δοκιμή-🔑"
    location = source(make_pdf("AES-256", user_password=password), "local")
    result = PdfReaderTool(PdfReaderToolConfig(password=password)).run(PdfReaderToolInputSchema(source=location))
    assert result.error is None
    assert result.page_count == 2
    assert "Second page content" in result.text


def test_page_range_validation_still_applies_after_decryption(source_factory):
    source, _ = source_factory
    result = PdfReaderTool(PdfReaderToolConfig(password=USER_PASSWORD)).run(
        PdfReaderToolInputSchema(source=source(make_pdf("RC4-128"), "local"), page_range="invalid")
    )
    assert "Invalid page number" in result.error
    assert result.pages == []
    assert result.metadata is None
