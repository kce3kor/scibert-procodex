import pytest

from scibert.inference import inference


@pytest.mark.inference
def test_inference():
    query = {
        "title": "This is the title",
        "content": "This is the content",
        "keywords": "This is a keywords",
    }

    assert type(inference(query=query)) == str

    assert inference(query) == "Relevant"
