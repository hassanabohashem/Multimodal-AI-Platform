"""Fixture tests for the official VQA accuracy metric — the easiest thing to get wrong."""
from mmp_training.vqa.metric import normalize_answer, vqa_accuracy


def test_normalization_articles_and_punctuation():
    assert normalize_answer("The dog!") == "dog"
    assert normalize_answer("a red car.") == "red car"


def test_normalization_number_words():
    assert normalize_answer("Two") == "2"
    assert normalize_answer("none") == "0"


def test_normalization_contractions():
    assert normalize_answer("dont") == "don't"


def test_accuracy_full_match():
    humans = ["yes"] * 10
    assert vqa_accuracy("Yes", humans) == 1.0


def test_accuracy_partial_match():
    humans = ["yes"] * 2 + ["no"] * 8
    assert abs(vqa_accuracy("yes", humans) - 2 / 3) < 1e-9


def test_accuracy_caps_at_one():
    humans = ["cat"] * 5 + ["dog"] * 5
    assert vqa_accuracy("cat", humans) == 1.0


def test_accuracy_zero():
    assert vqa_accuracy("bird", ["cat"] * 10) == 0.0
