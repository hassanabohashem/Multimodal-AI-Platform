"""Contract tests: unknown fields are rejected, constraints enforced."""
import pytest
from pydantic import ValidationError

from mmp_common.schemas import SearchRequest, VQARequest


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        SearchRequest(query="cat", top_k=5, surprise="field")


def test_top_k_bounds():
    with pytest.raises(ValidationError):
        SearchRequest(query="cat", top_k=500)


def test_question_length():
    with pytest.raises(ValidationError):
        VQARequest(question="")
