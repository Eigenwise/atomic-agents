import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from tool.hackernews_search import HackerNewsSearchTool, HackerNewsSearchToolConfig, HackerNewsSearchToolInputSchema


@pytest.fixture
def service():
    requests = []
    hits = [
        {"objectID": "1", "author": "Alice_1-test", "story_id": 100, "comment_text": "Matching comment"},
        {"objectID": "2", "author": "bob", "story_id": 100, "comment_text": "Other author's comment"},
        {"objectID": "3", "author": "Alice_1-test", "story_id": 200, "comment_text": "Other thread's comment"},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query, keep_blank_values=True)
            requests.append((url.path, params))
            tags = params["tags"][0].split(",")
            selected = hits
            for tag in tags:
                if tag.startswith("author_"):
                    selected = [hit for hit in selected if hit["author"] == tag[len("author_") :]]
                elif tag.startswith("story_"):
                    selected = [hit for hit in selected if str(hit["story_id"]) == tag[len("story_") :]]
            encoded = json.dumps({"hits": selected}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    tool = HackerNewsSearchTool(HackerNewsSearchToolConfig(base_url=f"http://127.0.0.1:{server.server_port}"))
    try:
        yield tool, requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("sort_by", ["relevance", "date"])
@pytest.mark.parametrize(
    "scope,expected_ids,expected_tags",
    [
        ({}, ["1", "2", "3"], "comment"),
        ({"author": "Alice_1-test"}, ["1", "3"], "comment,author_Alice_1-test"),
        ({"story_id": 100}, ["1", "2"], "comment,story_100"),
        ({"author": "Alice_1-test", "story_id": 100}, ["1"], "comment,author_Alice_1-test,story_100"),
    ],
)
async def test_filters_scope_each_query_before_results_are_returned(
    service, asynchronous, sort_by, scope, expected_ids, expected_tags
):
    tool, requests = service
    params = HackerNewsSearchToolInputSchema(
        queries=["café & agents", ""],
        tags="comment",
        sort_by=sort_by,
        max_results_per_query=5,
        numeric_filters="created_at_i>1700000000",
        **scope,
    )

    result = await tool.run_async(params) if asynchronous else tool.run(params)

    assert [item.object_id for item in result.results] == expected_ids * 2
    assert [item.query for item in result.results] == ["café & agents"] * len(expected_ids) + [""] * len(expected_ids)
    assert len(requests) == 2
    assert {request[1]["query"][0] for request in requests} == {"café & agents", ""}
    for path, query in requests:
        assert path == ("/search_by_date" if sort_by == "date" else "/search")
        assert query["tags"] == [expected_tags]
        assert query["numericFilters"] == ["created_at_i>1700000000"]
        assert query["hitsPerPage"] == ["5"]


def test_nonmatching_author_and_thread_returns_empty_results(service):
    tool, requests = service
    result = tool.run(HackerNewsSearchToolInputSchema(queries=[""], tags="comment", author="bob", story_id=200))

    assert result.results == []
    assert len(requests) == 1


@pytest.mark.parametrize("author", ["", "alice,bob", "alice,(story)", "alice bob", "alice\n"])
def test_author_cannot_add_filter_operators(author):
    with pytest.raises(ValidationError, match="author"):
        HackerNewsSearchToolInputSchema(queries=[""], author=author)


@pytest.mark.parametrize("story_id", [0, -1, True, "100,author_bob"])
def test_story_id_must_be_a_positive_integer(story_id):
    with pytest.raises(ValidationError, match="story_id"):
        HackerNewsSearchToolInputSchema(queries=[""], story_id=story_id)
