"""Tests for the ambiguity helpers: only real candidates may become choices."""

from oekg.disambiguation import (
    Option,
    build_user_prompt,
    candidate_options,
    clarification_hint,
    clarified_question,
    names_a_term,
    parse_clarification,
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
