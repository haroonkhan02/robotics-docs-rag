"""Download the corpus: the ROS 2 documentation sources (CC BY 4.0) at a pinned commit and
three open-access ROS 2 papers from arXiv (PDF ingestion path)."""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from roborag.config import load_config

ROS2_ZIP = "https://codeload.github.com/ros2/ros2_documentation/zip/{commit}"
# arXiv id -> short file stem. Downloaded for local indexing only; never redistributed.
PAPERS = {
    "2211.07752": "ros2-design-architecture-uses",  # Macenski et al., Science Robotics 2022
    "2003.00368": "marathon2-nav2",  # Macenski et al., IROS 2020
    "2307.15236": "nav2-algorithms-survey",  # Macenski et al., RAS 2023
}
PAPER_TITLES = {
    "ros2-design-architecture-uses":
        "Robot Operating System 2: Design, Architecture, and Uses In The Wild",
    "marathon2-nav2": "The Marathon 2: A Navigation System",
    "nav2-algorithms-survey": "From the Desks of ROS Maintainers: A Survey of Modern & Capable "
                              "Mobile Robotics Algorithms in the Robot Operating System 2",
}
UA = {"User-Agent": "robotics-docs-rag/1.0 (portfolio project)"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download_ros2_docs(data_dir: Path, commit: str, force: bool = False) -> Path:
    dest = data_dir / "ros2_documentation"
    if (dest / "source").exists() and not force:
        print(f"ROS 2 docs already in {dest}")
        return dest
    print(f"downloading ros2_documentation @ {commit[:10]} ...")
    blob = _get(ROS2_ZIP.format(commit=commit))
    tmp = data_dir / "_ros2_zip"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        # keep only the text sources and the licence, not the 40 MB of images
        members = [m for m in zf.namelist() if m.endswith((".rst", "LICENSE"))]
        zf.extractall(tmp, members)
    shutil.rmtree(dest, ignore_errors=True)
    next(tmp.iterdir()).rename(dest)
    tmp.rmdir()
    (dest / "COMMIT").write_text(commit + "\n")
    print(f"saved {len(members)} files to {dest}")
    return dest


def download_papers(pdf_dir: Path, force: bool = False) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for arxiv_id, stem in PAPERS.items():
        out = pdf_dir / f"{stem}.pdf"
        if out.exists() and not force:
            continue
        print(f"downloading arXiv:{arxiv_id} -> {out.name}")
        out.write_bytes(_get(f"https://arxiv.org/pdf/{arxiv_id}"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-pdfs", action="store_true", help="skip the arXiv papers")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    download_ros2_docs(data_dir, cfg["corpus"]["ros2_commit"], args.force)
    if not args.no_pdfs:
        download_papers(Path(cfg["corpus"]["pdf_dir"]), args.force)


if __name__ == "__main__":
    main()
