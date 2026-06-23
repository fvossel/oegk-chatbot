from oekg.cache import QueryCache, normalise_question


def test_normalise_question_collapses_and_lowercases():
    assert normalise_question("  Wie   VIELE?  ") == "wie viele?"


def test_put_get_roundtrip(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("How many scenarios?", "SELECT ...")
    entry = cache.get("how many   scenarios?")  # normalised match
    assert entry is not None
    assert entry.sparql == "SELECT ..."
    assert entry.hit_count == 0


def test_record_hit_increments_and_preserves_created_at(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("Q", "S")
    created = cache.get("Q").created_at
    cache.record_hit("Q")
    entry = cache.get("Q")
    assert entry.hit_count == 1
    assert entry.created_at == created


def test_corrupt_cache_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    cache = QueryCache(path)
    assert cache.get("anything") is None
    assert cache.list_questions() == []


def test_disabled_cache_is_inert(tmp_path):
    cache = QueryCache(tmp_path / "c.json", enabled=False)
    cache.put("Q", "S")
    assert cache.get("Q") is None
    assert cache.list_questions() == []


def test_clear_removes_all_entries(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("Q", "S")
    cache.clear()
    assert cache.list_questions() == []


def test_blank_sparql_is_not_stored(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("Q", "   ")
    assert cache.get("Q") is None


def test_embedding_roundtrip_and_has_embeddings(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    assert cache.has_embeddings() is False
    cache.put("Q", "S", embedding=[0.1, 0.2, 0.3])
    assert cache.has_embeddings() is True
    assert cache.get("Q").embedding == [0.1, 0.2, 0.3]


def test_put_preserves_existing_embedding(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("Q", "S", embedding=[1.0, 0.0])
    cache.put("Q", "S2")  # refresh sparql without re-supplying the embedding
    entry = cache.get("Q")
    assert entry.sparql == "S2"
    assert entry.embedding == [1.0, 0.0]


def test_semantic_match_returns_closest_above_threshold(tmp_path):
    cache = QueryCache(tmp_path / "c.json")
    cache.put("How many scenarios?", "SELECT a", embedding=[1.0, 0.0])
    cache.put("List the bundles", "SELECT b", embedding=[0.0, 1.0])

    match = cache.semantic_match([0.99, 0.01], threshold=0.9)
    assert match is not None
    entry, similarity = match
    assert entry.question == "How many scenarios?"
    assert similarity > 0.9

    # Nothing close enough -> no match.
    assert cache.semantic_match([0.7, 0.7], threshold=0.99) is None
