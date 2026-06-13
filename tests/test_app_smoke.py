"""Lightweight guards for the demo and docs (no Streamlit needed in CI)."""

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_compiles():
    """The Streamlit demo stays syntactically valid even without streamlit installed."""
    py_compile.compile(str(ROOT / "app.py"), doraise=True)


def test_docs_page_exists_and_references_assets():
    html = (ROOT / "docs" / "index.html").read_text()
    assert "driver_ablation_real.png" in html
    for asset in ["driver_ablation_real.png", "pr_curves_real.png"]:
        assert (ROOT / "docs" / "assets" / asset).exists()


def test_paper_present():
    paper = (ROOT / "PAPER.md").read_text()
    assert "## Abstract" in paper and "vegetation" in paper.lower()
