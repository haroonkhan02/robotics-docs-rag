import pytest
from fastapi.testclient import TestClient

from roborag.api import create_app
from roborag.generate import Generator, RAGPipeline
from roborag.retrieval import Retriever


@pytest.fixture
def client(chunks, fake_client):
    gen = Generator("claude-opus-5-5", client=fake_client("Call it with ros2 service call [1]."))
    return TestClient(create_app(RAGPipeline(Retriever(chunks), gen, "bm25", 3)))


def test_health_and_ui(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm_enabled"] and h["default_mode"] == "bm25"
    assert "ANTHROPIC" not in str(h).replace("llm_enabled", "")
    page = client.get("/")
    assert page.status_code == 200 and "Ask the ROS 2 docs" in page.text


def test_search(client):
    r = client.post("/search", json={"query": "ros2 service call", "k": 2}).json()
    assert len(r["results"]) <= 2 and r["results"][0]["rank"] == 1
    assert r["results"][0]["doc_id"] == "Tutorials/Services"


def test_ask_returns_answer_with_citations(client):
    r = client.post("/ask", json={"question": "How do I call a service?"}).json()
    assert r["citations"] == [1] and r["llm_used"] and len(r["passages"]) <= 3


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "x" * 1001},
                                     {"question": "ok", "mode": "magic"}, {"question": "ok",
                                                                             "k": 99}])
def test_input_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422
