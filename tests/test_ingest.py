from roborag.ingest import clean_inline, iter_rst_files, load_rst, parse_rst
from tests.conftest import SAMPLE_RST


def test_inline_markup_is_removed():
    s = "See :doc:`the guide <../x>` and ``ros2 run`` or `site <http://a.b>`_ **now** |DISTRO|"
    assert clean_inline(s) == "See the guide and ros2 run or site now DISTRO"


def test_headings_build_a_path_and_directives_are_handled():
    title, blocks = parse_rst(SAMPLE_RST)
    assert title == "Topics"
    headings = [(lvl, h) for lvl, h, _ in blocks if lvl]
    assert headings == [(1, "Topics"), (2, "Durability"), (3, "Echo a topic")]


def test_load_rst_keeps_text_and_code_but_drops_images_and_toctree(tmp_path):
    root = tmp_path / "source"
    (root / "Concepts").mkdir(parents=True)
    p = root / "Concepts" / "About-Topics.rst"
    p.write_text(SAMPLE_RST)
    doc = load_rst(p, root)
    assert doc.doc_id == "Concepts/About-Topics"
    assert doc.source.endswith("/Concepts/About-Topics.html")
    paths = [s.heading_path for s in doc.sections]
    assert ("Topics", "Durability", "Echo a topic") in paths
    text = doc.text
    assert "ros2 topic echo /turtle1/cmd_vel" in text  # code-block body kept
    assert "QoS guide" in text  # note body kept, role cleaned
    assert "late-joining" in text
    for gone in ("image::", "topic.gif", "Hidden-Page", "Table of Contents", "Old/Path", ":local:"):
        assert gone not in text


def test_exclude_globs(tmp_path):
    (tmp_path / "Releases").mkdir()
    (tmp_path / "Releases" / "Humble-Complete-Changelog.rst").write_text("x\n=\n\ny\n")
    (tmp_path / "Releases" / "Humble.rst").write_text("x\n=\n\ny\n")
    files = iter_rst_files(tmp_path, ["Releases/*Complete-Changelog*"])
    assert [f.name for f in files] == ["Humble.rst"]
