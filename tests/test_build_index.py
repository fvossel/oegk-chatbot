from oekg.build_index import _doc_text


def test_doc_text_combines_label_and_description():
    assert _doc_text({"label": "Author", "description": "An author."}) == "Author. An author."


def test_doc_text_handles_missing_description():
    assert _doc_text({"label": "Author"}) == "Author."
