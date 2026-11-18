"""The 6-prospect end-to-end simulation must land every prospect in its expected end state."""
from eval.simulate_prospects import simulate


def test_six_scripted_prospects(test_db):
    res = {r["prospect"]: r for r in simulate(test_db)}
    for name, r in res.items():
        assert r["ok"], r
    assert res["counter_then_confirms"]["proposals"] == 2          # a second proposal after the busy counter-offer
    assert res["ooo_then_interested"]["ooo_extensions"] == 1
    assert res["silent"]["crm_stage"] == "closed_no_reply"
