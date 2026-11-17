"""Exercise the RAG example's file-to-chunk path without provider requests."""

import importlib
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def chunk_document(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    return importlib.import_module("rag_chatbot.main").chunk_document


@pytest.mark.parametrize("text", ["0123456789" * 23, "\u6a21\u578b\u68c0\u7d22" * 53])
def test_long_paragraph_is_bounded_without_losing_text(tmp_path, chunk_document, text):
    document = tmp_path / "document.txt"
    document.write_text(text, encoding="utf-8")

    chunks = chunk_document(str(document), chunk_size=32, overlap=8)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 32 for chunk in chunks)
    assert all(right.startswith(left[-8:]) for left, right in zip(chunks, chunks[1:]))
    assert chunks[0] + "".join(chunk[8:] for chunk in chunks[1:]) == text


def test_paragraph_separators_count_toward_chunk_size(tmp_path, chunk_document):
    document = tmp_path / "document.txt"
    document.write_text("aaa\n\nbbb\n\nccc", encoding="utf-8")

    chunks = chunk_document(str(document), chunk_size=10, overlap=0)

    assert chunks == ["aaa\n\nbbb", "ccc"]


def test_overlap_uses_characters_instead_of_words(tmp_path, chunk_document):
    document = tmp_path / "document.txt"
    document.write_text("alpha beta gamma\n\ndelta epsilon zeta", encoding="utf-8")

    chunks = chunk_document(str(document), chunk_size=25, overlap=5)

    assert chunks[0] == "alpha beta gamma"
    assert chunks[1].startswith("gamma\n\ndelta")
    assert all(len(chunk) <= 25 for chunk in chunks)
    assert chunks[-1].endswith("zeta")


@pytest.mark.parametrize("text", ["", " \t\n\n \n"])
def test_empty_documents_produce_no_chunks(tmp_path, chunk_document, text):
    document = tmp_path / "document.txt"
    document.write_text(text, encoding="utf-8")
    assert chunk_document(str(document)) == []


@pytest.mark.parametrize("text", ["one short paragraph", "first paragraph\n\nsecond paragraph"])
def test_short_documents_keep_paragraphs(tmp_path, chunk_document, text):
    document = tmp_path / "document.txt"
    document.write_text(text, encoding="utf-8")
    assert chunk_document(str(document)) == [text]


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_chunk_settings_fail(tmp_path, chunk_document, chunk_size, overlap):
    document = tmp_path / "document.txt"
    document.write_text("some document", encoding="utf-8")
    with pytest.raises(ValueError):
        chunk_document(str(document), chunk_size=chunk_size, overlap=overlap)
