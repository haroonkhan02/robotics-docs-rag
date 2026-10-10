import pytest

from roborag.generate import ABSTAIN, Generator, RAGPipeline, build_messages, has_api_key
from roborag.retrieval import Hit, Retriever


def _hits(chunks, n=3):
    return [Hit(c, 1.0) for c in chunks[:n]]


def test_prompt_numbers_passages_with_sources(chunks):
    msg = build_messages("What is durability?", _hits(chunks))[0]["content"]
    assert msg.index("[1] ") < msg.index("[2] ") < msg.index("[3] ")
    assert "Source: https://x/topics" in msg and msg.rstrip().endswith("What is durability?")


def test_answer_parses_citations_and_request_shape(chunks, fake_client):
    client = fake_client("Use transient local durability [2]. Topics stream data [1][7].")
    gen = Generator("claude-opus-5-5", "low", 1000, client=client)
    ans = gen.answer("q?", _hits(chunks))
    assert ans.citations == [2, 1, 7] and ans.invalid_citations == [7]
    assert ans.llm_used and not ans.abstained and ans.usage["output_tokens"] == 45
    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default" and "thinking" not in call
    assert "square brackets" in call["system"]


def test_abstention_and_refusal(chunks, fake_client):
    gen = Generator(client=fake_client(ABSTAIN))
    assert gen.answer("q?", _hits(chunks)).abstained
    refused = Generator(client=fake_client("", stop_reason="refusal")).answer("q?", _hits(chunks))
    assert refused.stop_reason == "refusal" and "declined" in refused.text


def test_without_key_the_pipeline_still_returns_passages(chunks, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not has_api_key()
    gen = Generator()
    assert not gen.enabled
    pipe = RAGPipeline(Retriever(chunks), gen, mode="bm25", top_k=2)
    ans = pipe.ask("ros2 service call")
    assert not ans.llm_used and len(ans.hits) == 2 and "retrieval_ms" in ans.usage
    d = ans.to_dict()
    assert d["passages"][0]["n"] == 1 and "text" in d["passages"][0]


def test_env_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=test-not-a-real-key\n")
    try:
        assert has_api_key()
    finally:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.mark.parametrize("hits", [[]])
def test_no_passages_means_abstain(hits, fake_client):
    ans = Generator(client=fake_client("unused")).answer("q?", hits)
    assert ans.abstained and ans.text == ABSTAIN
