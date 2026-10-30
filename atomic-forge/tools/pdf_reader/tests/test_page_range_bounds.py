from multiprocessing import get_context

from tool.pdf_reader import PdfReaderTool


def _select_large_ranges(connection):
    with connection:
        connection.send("ready")
        end = 10**30
        results = [
            PdfReaderTool.parse_page_range(f"1-{end}", 3),
            PdfReaderTool.parse_page_range(f"2-{end},1", 3),
            PdfReaderTool.parse_page_range(f"4-{end},2", 3),
        ]
        for expression, total_pages in [(f"4-{end}", 3), (f"1-{end}", 0)]:
            try:
                PdfReaderTool.parse_page_range(expression, total_pages)
            except ValueError as error:
                results.append("No pages selected" in str(error))
            else:
                results.append(False)
        connection.send(results)


def test_large_ranges_are_bounded_by_the_document_page_count():
    # Isolate the call so a regression fails without hanging the test runner.
    context = get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_select_large_ranges, args=(child,))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Page-selection worker failed to start"
        assert parent.recv() == "ready"
        assert parent.poll(5), "Page selection iterated beyond the document's page count"
        assert parent.recv() == [[1, 2, 3], [1, 2, 3], [2], True, True]
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
