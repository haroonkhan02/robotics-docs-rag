from pathlib import Path

from roborag.chunking import Chunk
from roborag.evalset import Question, check_questions, is_relevant, load_questions, norm

ROOT = Path(__file__).resolve().parents[1]


def _chunk(doc_id, text):
    return Chunk("id", doc_id, "t", "s", text, "src")


def test_relevance_needs_gold_doc_and_key():
    q = Question("q1", "cli", "?", ["A"], ["ros2 topic   hz"])
    assert is_relevant(_chunk("A", "run ROS2 TOPIC hz\n/x"), q)  # case + whitespace insensitive
    assert not is_relevant(_chunk("B", "ros2 topic hz"), q)  # wrong document
    assert not is_relevant(_chunk("A", "ros2 topic echo"), q)  # right doc, no answer
    assert norm("std::unique_ptr\\ s") == "std::unique_ptr s"


def test_check_questions_reports_problems():
    docs = {"A": "the answer is 42"}
    qs = [Question("q1", "c", "?", ["A"], ["answer is 42"]),
          Question("q2", "c", "?", ["A"], ["missing phrase"]),
          Question("q3", "c", "?", ["Nope"], ["42"]),
          Question("u1", "unanswerable", "?")]
    problems = check_questions(qs, docs)
    assert len(problems) == 3 and problems[0].startswith("q2")


def test_shipped_eval_set_is_well_formed():
    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    assert len(qs) >= 60
    assert len({q.id for q in qs}) == len(qs)
    for q in qs:
        assert q.question.endswith("?") and q.reference_answer
        if q.answerable:
            assert q.answer_keys and all(k.strip() for k in q.answer_keys)
    assert sum(not q.answerable for q in qs) >= 5
