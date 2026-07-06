"""Answer-type routing heuristic used by the VQA endpoint."""
from mmp_gateway.routers.vqa import classify_question


def test_yes_no():
    assert classify_question("Is there a dog in the image?") == "yes/no"


def test_number():
    assert classify_question("How many people are visible?") == "number"


def test_other():
    assert classify_question("What color is the car?") == "other"
