from roborag.evalset import Question
from roborag.evaluate_answers import JUDGE_SCHEMA, aggregate, judge, run, score_item
from roborag.generate import Generator, RAGPipeline
from roborag.retrieval import Retriever

VERDICT = {
    "claims": [
        {"claim": "a", "cited": [1], "supported_by_cited": True, "supported_by_context": True},
        {"claim": "b", "cited": [2], "supported_by_cited": False, "supported_by_context": True},
        {"claim": "c", "cited": [], "supported_by_cited": False, "supported_by_context": False},
    ],
    "abstained": False, "correctness": "partially_correct", "explanation": "x",
}
Q = Question("q1", "cli", "How do I use ros2 service call?", ["A"], ["k"], "ref")
U = Question("u1", "unanswerable", "What?")


def test_score_item():
    s = score_item(Q, VERDICT)
    assert s["faithfulness"] == 2 / 3
    assert s["citation_accuracy"] == 1 / 2  # only claims that cite something
    assert s["uncited_rate"] == 1 / 3 and s["correctness"] == 0.5 and s["n_claims"] == 3
    assert score_item(U, {**VERDICT, "abstained": True})["correctness"] is None


def test_aggregate_abstention_rates():
    base = {"faithfulness": 1.0, "citation_accuracy": 1.0, "uncited_rate": 0.0,
            "correctness": 1.0, "invalid_citations": [], "latency_ms": 10.0}
    items = [{**base, "answerable": True, "abstained": False},
             {**base, "answerable": True, "abstained": True, "correctness": None},
             {**base, "answerable": False, "abstained": True, "correctness": None}]
    agg = aggregate(items)
    assert agg["false_abstention_rate"] == 0.5 and agg["correct_abstention_rate"] == 1.0
    assert agg["correctness"] == 1.0 and agg["n_unanswerable"] == 1


def test_judge_uses_structured_output(fake_client, chunks):
    client = fake_client(judge=VERDICT)
    gen = Generator(client=fake_client("Answer [1]."))
    ans = RAGPipeline(Retriever(chunks), gen, "bm25", 2).ask("ros2 service call")
    verdict = judge(client, "claude-opus-5-5", Q, ans)
    assert verdict["correctness"] == "partially_correct"
    call = client.messages.calls[0]
    assert call["output_config"]["format"]["schema"] is JUDGE_SCHEMA
    assert "<reference_answer>\nref" in call["messages"][0]["content"]


def test_run_end_to_end_with_mocks(fake_client, chunks):
    pipe = RAGPipeline(Retriever(chunks), Generator(client=fake_client("Answer [1].")), "bm25", 2)
    out = run(pipe, fake_client(judge=VERDICT), "m", [Q, U])
    assert len(out["items"]) == 2 and out["summary"]["n_answerable"] == 1
    assert out["items"][0]["citations"] == [1]
    # nothing retrieved for the unanswerable question -> abstain without calling the LLM
    assert out["items"][1]["answer"].startswith("I can't answer")
