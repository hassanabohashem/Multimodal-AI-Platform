"""Official VQA v2 accuracy metric with answer normalization.

acc(ans) = min(#humans_who_gave_ans / 3, 1), averaged over all 10
leave-one-out subsets — equivalently computed against the full 10 answers.
Public reimplementations frequently get the normalization wrong; this module
is covered by fixture tests in tests/unit/test_vqa_metric.py.
"""
from __future__ import annotations

import re

_CONTRACTIONS = {
    "arent": "aren't", "cant": "can't", "couldnt": "couldn't", "didnt": "didn't",
    "doesnt": "doesn't", "dont": "don't", "hasnt": "hasn't", "havent": "haven't",
    "isnt": "isn't", "shouldnt": "shouldn't", "wasnt": "wasn't", "werent": "weren't",
    "wont": "won't", "wouldnt": "wouldn't", "youre": "you're", "theyre": "they're",
}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.]")


def normalize_answer(answer: str) -> str:
    """Apply the official VQA normalization: punctuation, articles, numbers, contractions."""
    s = answer.lower().strip()
    s = _PUNCT.sub("", s)
    s = s.replace(":", "")
    words = []
    for w in s.split():
        w = _NUMBER_WORDS.get(w, w)
        w = _CONTRACTIONS.get(w, w)
        if w not in _ARTICLES:
            words.append(w)
    return " ".join(words)


def vqa_accuracy(prediction: str, human_answers: list[str]) -> float:
    """Score one prediction against the 10 human answers."""
    pred = normalize_answer(prediction)
    humans = [normalize_answer(a) for a in human_answers]
    matches = sum(1 for h in humans if h == pred)
    return min(matches / 3.0, 1.0)
