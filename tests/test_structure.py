"""Grounding: a fake quote → field stays null; value must be in its own quote; never overwritten by null."""
from agent.llm import LLM, FakeProvider
from agent.models import PageRecord
from pipeline.structure import Claim, merge, quote_match, structure, verify

HOME = "https://acme.io/"
ABOUT = "https://acme.io/about"
PAGES = [
    PageRecord(url=HOME, page_type="home", clean_text=(
        "## HERO\nAcme is the compliance-ready billing platform for B2B SaaS finance teams.\n"
        "## FOOTER BADGES\nWe are currently working towards SOC 2 Type II and are GDPR compliant today.")),
    PageRecord(url=ABOUT, page_type="about", clean_text=(
        "## ABOUT\nIn March we raised a $30M Series B led by Accel to expand into Europe.\n"
        "Today we are a team of 140 people across Bangalore, London and New York.")),
]


def c(field, value, quote, url=HOME, status="n/a"):
    return Claim(field=field, value=value, quote=quote, source_url=url, cert_status=status)


def test_exact_and_fuzzy_quote_match():
    assert quote_match("the compliance-ready billing platform for B2B SaaS", PAGES[0].clean_text) == 1.0
    # curly quotes / punctuation / spacing differences are tolerated
    assert quote_match("Acme is the compliance ready billing platform, for B2B SaaS finance", PAGES[0].clean_text) >= 0.9


def test_invented_quote_is_rejected_and_field_stays_null():
    fake = c("funding_stage", "Series C", "we closed a 50 million dollar Series C round", ABOUT)
    good, rejected = verify([fake], PAGES)
    assert good == [] and rejected[0].field == "funding_stage" and "not found" in rejected[0].reason
    assert merge(None, good).funding_stage is None


def test_real_quote_but_invented_value_rejected():
    # quote is real, but it says Series B — the model wrote Series C
    lie = c("funding_stage", "Series C", "we raised a $30M Series B led by Accel", ABOUT)
    _, rejected = verify([lie], PAGES)
    assert "does not appear in its own quote" in rejected[0].reason


def test_stage_never_inferred_from_headcount():
    infer = c("funding_stage", "Series B", "Today we are a team of 140 people across Bangalore", ABOUT)
    good, rejected = verify([infer], PAGES)
    assert not good and rejected


def test_url_must_be_fetched_and_quote_length():
    _, r1 = verify([c("buyer", "finance teams", "billing platform for B2B SaaS finance teams", "https://acme.io/blog")], PAGES)
    _, r2 = verify([c("buyer", "finance teams", "finance teams", HOME)], PAGES)
    assert "not a page we fetched" in r1[0].reason and "words" in r2[0].reason


def test_claimed_downgraded_when_quote_says_working_towards():
    good, _ = verify([c("certification", "SOC 2", "We are currently working towards SOC 2 Type II", status="claimed")], PAGES)
    assert good[0].cert_status == "in_progress"


def test_good_claims_assemble_into_company_facts():
    claims = [
        c("is_b2b_saas", "true", "the compliance-ready billing platform for B2B SaaS finance teams"),
        c("funding_stage", "Series B", "we raised a $30M Series B led by Accel", ABOUT),
        c("headcount", "140", "Today we are a team of 140 people across Bangalore", ABOUT),
        c("certification", "GDPR", "SOC 2 Type II and are GDPR compliant today", status="claimed"),
        c("region", "Europe", "led by Accel to expand into Europe", ABOUT),
    ]
    good, rejected = verify(claims, PAGES)
    assert not rejected
    f = merge(None, good)
    assert f.is_b2b_saas.value is True and f.funding_stage.value == "Series B" and f.headcount.value == 140
    assert f.certifications[0].name == "GDPR" and f.funding_stage.source_url == ABOUT


def test_verified_fact_never_overwritten_and_cert_upgrades():
    first = merge(None, [c("funding_stage", "Series B", "we raised a $30M Series B led by Accel", ABOUT),
                         c("certification", "SOC 2", "working towards SOC 2 Type II and are GDPR", status="in_progress")])
    second = merge(first, [])                                  # a later pass that found nothing
    assert second.funding_stage.value == "Series B"
    third = merge(second, [c("certification", "SOC 2", "x" * 5, status="claimed")])
    assert third.certifications[0].status == "claimed"
    assert merge(third, [c("certification", "SOC 2", "y", status="mentioned")]).certifications[0].status == "claimed"


def test_structure_end_to_end_with_fake_llm_and_db(test_db):
    reply = {"claims": [
        {"field": "funding_stage", "value": "Series B", "quote": "we raised a $30M Series B led by Accel", "source_url": ABOUT},
        {"field": "headcount", "value": "999", "quote": "a team of 999 brilliant humans", "source_url": ABOUT},
    ]}
    fake = FakeProvider([reply])
    test_db.execute("INSERT INTO companies (id, domain) VALUES (1, 'acme.io')")
    facts, rejected, out = structure(LLM(fake), "acme.io", PAGES, conn=test_db, company_id=1)
    assert out.ok and facts.funding_stage.value == "Series B" and facts.headcount is None
    assert len(rejected) == 1
    rows = test_db.execute("SELECT field, verified FROM facts ORDER BY id").fetchall()
    assert rows == [{"field": "funding_stage", "verified": True}, {"field": "headcount", "verified": False}]
    prompt = fake.prompts[0][1]
    assert "=== PAGE: https://acme.io/about" in prompt                     # all pages so far are sent


def test_structure_invalid_json_twice_fails_company():
    facts, _, out = structure(LLM(FakeProvider(["nope", "nope"])), "acme.io", PAGES, sleep=lambda s: None)
    assert not out.ok and out.action == "fail_company"
