"""Synthetic fixtures: a tiny RST corpus, a hashing embedder and a fake Claude client.

Nothing here downloads a model, needs a GPU, or calls the network.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from roborag.chunking import chunk_corpus
from roborag.ingest import Document, Section
from roborag.retrieval import TOKEN_RE

SAMPLE_RST = """\
.. redirect-from::

    Old/Path

.. _Topics:

Topics
======

.. contents:: Table of Contents
   :local:

Topics are one of the three primary styles of interfaces provided by ROS 2.

.. image:: images/topic.gif

Durability
----------

* Transient local: the publisher becomes responsible for persisting samples for "late-joining"
  subscriptions.

.. note:: See the :doc:`QoS guide <../About-QoS>` for ``ros2 topic echo`` details.

Echo a topic
^^^^^^^^^^^^

Run the following command:

.. code-block:: console

   $ ros2 topic echo /turtle1/cmd_vel

.. toctree::
   :hidden:

   Hidden-Page
"""


class HashEmbedder:
    """Deterministic bag-of-words hashing 'embedding': shared words -> high cosine."""

    model_name = "hash-embedder"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts, progress=False):
        return np.stack([self._vec(t) for t in texts])

    def encode_queries(self, texts):
        return self.encode_docs(texts)


class LengthReranker:
    """Fake cross-encoder: prefers the *shortest* passage, so its effect is easy to see."""

    def score(self, query, texts):
        return [-float(len(t)) for t in texts]


def make_docs() -> list[Document]:
    return [
        Document("Concepts/Topics", "Topics", "https://x/topics", "rst", [
            Section(("Topics",), "Topics carry continuous data streams such as sensor data."),
            Section(("Topics", "Durability"),
                    "Transient local durability keeps samples for late-joining subscriptions."),
        ]),
        Document("Tutorials/Services", "Services", "https://x/services", "rst", [
            Section(("Services",), "Services are request response calls that return quickly."),
            Section(("Services", "Call"), "Use ros2 service call to call a service from the CLI."),
        ]),
        Document("Tutorials/Bags", "Bags", "https://x/bags", "rst", [
            Section(("Bags",), "Record topics with ros2 bag record -o name and replay them."),
        ]),
        Document("pdf/paper", "A Paper", "data/pdfs/paper.pdf", "pdf", [
            Section(("A Paper", "page 1"), "Navigation2 uses a behavior tree.", page=1),
            Section(("A Paper", "page 2"), "The robots drove 37.4 miles.", page=2),
        ]),
    ]


@pytest.fixture
def docs():
    return make_docs()


@pytest.fixture
def chunks(docs):
    return chunk_corpus(docs, "section", max_words=60)


@pytest.fixture
def embedder():
    return HashEmbedder()


class FakeMessages:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            model=kwargs.get("model"),
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
        )


class FakeClient:
    """Mimics the parts of anthropic.Anthropic the project uses (messages + beta.messages)."""

    def __init__(self, text: str = "", stop_reason: str = "end_turn", judge: dict | None = None):
        self.beta = SimpleNamespace(messages=FakeMessages(text, stop_reason))
        self.messages = FakeMessages(json.dumps(judge or {}))


@pytest.fixture
def fake_client():
    return FakeClient
