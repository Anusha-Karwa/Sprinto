from agent.config import icp, outreach_cfg, poc_overrides, sprinto_facts


def test_icp_numbers_from_config():
    c = icp()
    assert c["loop"]["max_extra_fetches"] == 4
    assert c["thresholds"] == {"qualified": 60, "flagged_floor": 40}
    r = c["rubric"]
    assert r["stage"]["named_a_to_c"] + r["pressure"]["cap"] + r["expansion_gap"] + r["b2b_clarity"] == 100


def test_sprinto_facts_proof_points_empty_until_verified():
    f = sprinto_facts()
    assert f["proof_points"] == [] and f["capabilities"] and f["cta_options"]


def test_outreach_caps():
    assert outreach_cfg()["sequence2"]["max_reply_rounds"] == 2


def test_poc_overrides_ignores_comment():
    assert all(not k.startswith("_") for k in poc_overrides())
