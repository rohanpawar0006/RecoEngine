"""
Smoke test for Streamlit demo application.
Ensures demo/app.py components and helpers execute without error.
"""

from pathlib import Path

from demo.app import get_genre_border_color, load_all_reports


def test_genre_border_colors():
    """Verify genre accent color mapping."""
    assert get_genre_border_color("Action|Adventure") == "#FF6B6B"
    assert get_genre_border_color("Comedy") == "#FFB84D"
    assert get_genre_border_color("Drama") == "#8B7CFF"
    assert get_genre_border_color("Sci-Fi") == "#4C8DFF"
    assert get_genre_border_color("UnknownGenre") == "#9AA4B0"


def test_load_all_reports():
    """Verify demo loader reads all generated reports."""
    reports = load_all_reports()
    assert "final_results.json" in reports
    assert "serving_latency.json" in reports
    assert "retraining_log.json" in reports
    assert reports["final_results.json"]["metadata"]["dataset"] == "MovieLens 25M (temporal 80/10/10 split)"


def test_demo_file_exists():
    """Verify demo/app.py is present and non-empty."""
    app_path = Path(__file__).resolve().parent.parent / "demo" / "app.py"
    assert app_path.exists()
    assert app_path.stat().st_size > 1000

