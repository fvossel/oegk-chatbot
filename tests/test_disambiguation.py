"""Tests for the ambiguity helpers: only real candidates may become choices."""

from oekg.disambiguation import (
    Option,
    alias_hint,
    build_user_prompt,
    candidate_options,
    clarification_hint,
    clarified_question,
    names_a_term,
    parse_clarification,
    term_mentions,
)

A = Option("oeo:A", "scenario bundle", "A collection of scenarios.")
B = Option("oeo:B", "scenario factsheet", "Describes one scenario.")
C = Option("oeo:C", "author", "Writes things.")
CANDIDATES = [A, B, C]


def _parse(reply, max_options=4):
    choice = parse_clarification(reply, CANDIDATES, max_options=max_options)
    return list(choice.options) if choice else []


def test_two_real_candidates_become_options():
    assert _parse('{"ambiguous": true, "options": ["C1", "C2"]}') == [A, B]


def test_a_clear_question_yields_nothing():
    assert _parse('{"ambiguous": false}') == []
    assert _parse('{"ambiguous": false, "options": ["C1", "C2"]}') == []


def test_invented_identifiers_are_dropped_and_one_option_is_not_a_choice():
    assert _parse('{"ambiguous": true, "options": ["C1", "C9"]}') == []
    assert _parse('{"ambiguous": true, "options": ["C1", "C2", "C9"]}') == [A, B]


def test_duplicates_collapse_and_the_cap_applies():
    assert _parse('{"ambiguous": true, "options": ["C1", "C1"]}') == []
    assert _parse('{"ambiguous": true, "options": ["C1", "C2", "C3"]}', max_options=2) == [A, B]


def test_ids_are_matched_leniently_but_only_to_real_candidates():
    assert _parse('{"ambiguous": true, "options": [" c1 ", "c3"]}') == [A, C]


def test_malformed_replies_never_ask():
    for reply in ("", "yes", "{", "[1, 2]", '{"ambiguous": "true", "options": ["C1","C2"]}',
                  '{"ambiguous": true, "options": "C1, C2"}', '{"ambiguous": true}'):
        assert _parse(reply) == [], reply


def test_fenced_or_chatty_json_is_still_read():
    fenced = '```json\n{"ambiguous": true, "options": ["C2", "C3"]}\n```'
    assert _parse(fenced) == [B, C]
    assert _parse('Sure! {"ambiguous": true, "options": ["C1", "C3"]} Hope that helps.') == [A, C]


def test_candidates_are_the_first_classes_only():
    docs = [
        {"uri": "oeo:rel", "label": "has part"},
        {"class": "oeo:A", "label": "scenario bundle", "description": "x"},
        {"class": "oeo:B", "label": "scenario factsheet", "description": "y"},
        {"class": "oeo:C", "label": "author", "description": "z"},
        "not a dict",
    ]
    assert [o.uri for o in candidate_options(docs, 2)] == ["oeo:A", "oeo:B"]


def test_long_definitions_are_shortened_at_a_word_boundary():
    long = " ".join(["word"] * 200)
    (option,) = candidate_options([{"class": "oeo:A", "label": "a", "description": long}], 1)
    assert len(option.definition) < 340 and option.definition.endswith("…")


def test_the_prompt_numbers_the_candidates_to_match_the_parser():
    prompt = build_user_prompt("Q?", CANDIDATES)
    assert "C1: scenario bundle" in prompt and "C3: author" in prompt and "Q?" in prompt


def test_the_prompt_lists_the_everyday_names_of_a_candidate():
    named = Option("oeo:B", "scenario factsheet", "Describes one scenario.", ("scenario",))
    assert "C1: scenario factsheet (also called: scenario) -- " in build_user_prompt("Q?", [named])


def test_the_clarified_question_and_hint_carry_the_choice():
    assert clarified_question("How many?", A) == "How many? (scenario bundle)"
    hint = clarification_hint(A)
    assert A.uri in hint and A.definition in hint


def test_a_question_using_a_terms_own_name_needs_no_clarification():
    for question in (
        "Which scenario bundles use MESSAGEix?",
        "How many scenario factsheets exist?",
        "List all authors.",
        "SHOW THE SCENARIO-BUNDLE LIST",
    ):
        assert names_a_term(question, CANDIDATES), question


def test_a_vaguer_question_is_not_treated_as_naming_a_term():
    for question in (
        "How many scenarios are there?",
        "Who made the scenarios?",
        "Which bundles use MESSAGEix?",
        "List the autho",  # a fragment of a word is not the word
    ):
        assert not names_a_term(question, CANDIDATES), question


def test_a_term_inside_a_longer_word_does_not_count():
    assert not names_a_term("Which coauthors collaborated?", CANDIDATES)


def _choice(reply):
    return parse_clarification(reply, CANDIDATES, max_options=4)


def test_the_words_around_the_choice_are_kept_in_the_users_language():
    choice = _choice(
        '{"ambiguous": true, "options": ["C1", "C2"], '
        '"ask": "Welchen Begriff meinen Sie?", "skip": "Keiner, entscheide du"}'
    )
    assert choice.options == (A, B)
    assert choice.ask == "Welchen Begriff meinen Sie?"
    assert choice.skip == "Keiner, entscheide du"


def test_missing_or_unusable_wording_falls_back_to_empty():
    for reply in (
        '{"ambiguous": true, "options": ["C1", "C2"]}',
        '{"ambiguous": true, "options": ["C1", "C2"], "ask": 5, "skip": ["x"]}',
        '{"ambiguous": true, "options": ["C1", "C2"], "ask": "' + "x" * 500 + '"}',
    ):
        choice = _choice(reply)
        assert choice.options == (A, B), reply
        assert choice.ask == "" and choice.skip == "", reply


def test_wording_is_flattened_to_one_line():
    choice = _choice('{"ambiguous": true, "options": ["C1", "C2"], "ask": "Which\\n  one?"}')
    assert choice.ask == "Which one?"


def test_wording_never_makes_a_clear_question_ambiguous():
    assert _choice('{"ambiguous": false, "ask": "Which one?", "skip": "No"}') is None


# -- everyday names (aliases) ---------------------------------------------------

BUNDLE = Option("oeo:A", "scenario bundle", "A collection of scenarios.", ("bundles", "studies", "studien"))
FACTSHEET = Option("oeo:B", "scenario factsheet", "Describes one scenario.", ("scenarios", "szenarien"))
REPORT = Option("oeo:R", "study report", "A report from a study.")
REGION = Option("oeo:G", "study region", "The region a study covers.")
TERMS = [BUNDLE, FACTSHEET, REPORT, REGION]


def _mentioned(question):
    return [(m.option.label, m.words, m.by_alias) for m in term_mentions(question, TERMS)]


def test_aliases_are_read_from_the_class_documents():
    doc = {"class": "oeo:B", "label": "scenario factsheet", "description": "y", "aliases": ["scenario", " "]}
    (option,) = candidate_options([doc], 1)
    assert option.aliases == ("scenario",)


def test_an_everyday_name_names_its_term():
    for question, label in (
        ("How many scenarios are there?", "scenario factsheet"),
        ("Which bundles use MESSAGEix?", "scenario bundle"),
        ("Which studies cover Germany?", "scenario bundle"),
        ("Wie viele Szenarien gibt es?", "scenario factsheet"),
        ("Welche Studien gibt es?", "scenario bundle"),
    ):
        assert [m[0] for m in _mentioned(question)] == [label], question
        assert names_a_term(question, [BUNDLE, FACTSHEET]), question


def test_the_longest_name_wins_and_owns_its_words():
    # "scenario bundles" is the bundle, not also a factsheet called "scenario".
    assert _mentioned("How many scenario bundles are there?") == [("scenario bundle", "scenario bundles", False)]
    # "study reports" and "study regions" are their own terms, not studies (bundles).
    assert _mentioned("List the study reports.") == [("study report", "study reports", False)]
    assert _mentioned("Which study regions exist?") == [("study region", "study regions", False)]


def test_a_longer_name_outside_the_candidates_still_blocks_an_alias():
    # The candidates are bundle and factsheet only; "study reports" must not read as "studies".
    assert not names_a_term("List the study reports.", [BUNDLE, FACTSHEET], vocabulary=TERMS)


def test_a_condition_term_does_not_hide_the_counted_term():
    mentioned = _mentioned("How many bundles have at least one scenario assigned?")
    assert [m[0] for m in mentioned] == ["scenario bundle", "scenario factsheet"]


def test_the_alias_hint_tells_generation_what_an_everyday_name_means():
    hint = alias_hint("How many scenarios are there?", TERMS)
    assert '"scenarios" means scenario factsheet (oeo:B)' in hint  # the user's own word
    assert '"study" means scenario bundle (oeo:A)' in alias_hint("Which study used it?", TERMS)
    assert alias_hint("How many scenario bundles are there?", TERMS) == ""  # named by label
    assert alias_hint("Who wrote it?", TERMS) == ""


def test_plurals_fold_onto_singulars():
    assert names_a_term("Which studies exist?", [BUNDLE])
    assert names_a_term("Which bundles exist?", [BUNDLE])


# -- "all of these" and plain-words options --------------------------------------


def test_all_options_can_be_chosen_together():
    assert clarified_question("How many?", (A, B)) == "How many? (scenario bundle, scenario factsheet)"
    hint = clarification_hint([A, B])
    assert A.uri in hint and B.uri in hint and "separately" in hint
    assert clarified_question("How many?", (A,)) == clarified_question("How many?", A)


def test_options_may_carry_a_plain_words_gloss_and_an_all_label():
    choice = _choice(
        '{"ambiguous": true, "options": [{"id": "C1", "gloss": "Studien mit mehreren Szenarien"}, '
        '{"id": "c2", "gloss": "einzelne Szenarien"}], "all": "Alle davon", "skip": "Keiner"}'
    )
    assert [o.uri for o in choice.options] == [A.uri, B.uri]
    assert [o.gloss for o in choice.options] == ["Studien mit mehreren Szenarien", "einzelne Szenarien"]
    assert choice.all == "Alle davon" and choice.skip == "Keiner"


def test_a_missing_or_unusable_gloss_falls_back_to_empty():
    choice = _choice(
        '{"ambiguous": true, "options": [{"id": "C1"}, {"id": "C2", "gloss": ["x"]}, "C3"]}'
    )
    assert [o.gloss for o in choice.options] == ["", "", ""]
    assert choice.all == ""


def test_an_invented_id_inside_an_object_is_still_dropped():
    assert _parse('{"ambiguous": true, "options": [{"id": "C1"}, {"id": "C9"}]}') == []
