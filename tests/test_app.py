"""The Review Queue renders a pending draft next to its evidence (Streamlit AppTest, test DB)."""
import os

import pytest


def test_review_queue_renders(test_db, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from agent import config
    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    config.settings.cache_clear()
    from tests.test_feedback import mk_draft
    mk_draft(test_db)
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    assert any("Review Queue" in h.value for h in at.header)
    assert any("Ajay" in s.value for s in at.subheader)
    assert [b.label for b in at.button][:3] == ["Approve as-is", "Approve with edits", "Reject (rewrite)"]
    at.sidebar.radio[0].set_value("Questions for you").run()
    assert not at.exception
    config.settings.cache_clear()
