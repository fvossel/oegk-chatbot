import pytest

from oekg.config import get_config
from oekg.resources import _validate, load_resources


def test_load_resources_is_internally_consistent():
    res = load_resources(get_config())
    assert res.faiss_index.ntotal == len(res.ids)
    assert len(res.relation_ids) > 0
    assert all(doc_id in res.documents_dict for doc_id in res.ids)


def test_validate_raises_on_size_mismatch():
    class _Index:
        ntotal = 3

    with pytest.raises(ValueError, match="does not match"):
        _validate(_Index(), ["a", "b"], {"a": {}, "b": {}})


def test_validate_raises_on_missing_document():
    class _Index:
        ntotal = 2

    with pytest.raises(ValueError, match="absent"):
        _validate(_Index(), ["a", "b"], {"a": {}})
