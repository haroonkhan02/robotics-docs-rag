import math

import pytest

from roborag.metrics import (
    bootstrap_ci,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_p,
    parse_citations,
    precision_at_k,
)

RELS = [False, False, True, False, True]


def test_hit_mrr_precision():
    assert hit_at_k(RELS, 2) == 0.0 and hit_at_k(RELS, 3) == 1.0
    assert mrr_at_k(RELS, 5) == pytest.approx(1 / 3)
    assert mrr_at_k(RELS, 2) == 0.0
    assert precision_at_k(RELS, 5) == pytest.approx(0.4)


def test_ndcg_worked_example():
    dcg = 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RELS, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([True, True], 5) == pytest.approx(1.0)
    # a third relevant chunk exists in the index but was not retrieved -> lower ideal match
    assert ndcg_at_k([True, True], 5, n_relevant=3) < 1.0
    assert ndcg_at_k([False] * 5, 5) == 0.0


def test_bootstrap_ci_contains_mean_and_is_degenerate_for_constants():
    vals = [0.0, 1.0] * 30
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.5 < hi
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_paired_bootstrap():
    a = [1.0] * 40
    assert paired_bootstrap_p(a, a) == 1.0
    b = [0.0] * 40
    assert paired_bootstrap_p(a, b) < 0.01
    with pytest.raises(ValueError):
        paired_bootstrap_p(a, b[:3])


def test_parse_citations():
    assert parse_citations("Use DDS [1][3]. Also [2, 4] and again [1].") == [1, 3, 2, 4]
    assert parse_citations("no citations here, list[int] is code") == []
