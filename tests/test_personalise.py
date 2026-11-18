"""Pain points must cite verified quotes; drafts are linted; send_email can only reach MY_EMAIL."""
import base64
import email

import pytest

from agent.llm import LLM, FakeProvider
from agent.models import BoolFact, CompanyFacts, Contact, EmailDraft, StrFact, VerdictResult
from pipeline.personalise import draft_email, lint, pain_points

U = "https://acme.io/pricing"
Q_SSO = "Enterprise plan includes SSO, SAML and audit logs"
Q_HIRE = "We are hiring a Security Compliance Manager to lead our audits"
FACTS = CompanyFacts(company_name=StrFact(value="Acme", quote="Acme is billing software for finance teams", source_url=U),
                     is_b2b_saas=BoolFact(value=True, quote="Acme is billing software for finance teams", source_url=U),
                     enterprise_signals=[StrFact(value="SSO", quote=Q_SSO, source_url=U)],
                     compliance_hiring=[StrFact(value="Security Compliance Manager", quote=Q_HIRE, source_url=U)])
VIK = Contact(first_name="Vik", last_name="Iyer", position="CTO", role_bucket="cto", confidence=92)
V = VerdictResult(verdict="QUALIFIED", score=80, reasons=["Pressure +15: SSO"])
CTA = "Worth a short conversation?"
CAP = "Sprinto automates evidence collection by integrating with your cloud, HR and code tools."
GOOD_BODY = (f"Hi Vik,\n\nSaw that Acme's enterprise plan includes SSO, SAML and audit logs. Enterprise buyers who want "
             f"those usually also ask for a SOC 2 report, and the evidence work lands on engineering.\n\n{CAP}\n\n{CTA}\n\nAnu, Sprinto")


def good(**kw):
    d = dict(subject="SSO buyers and SOC 2", body=GOOD_BODY, hook_quote=Q_SSO, capability_id="evidence_automation", cta=CTA)
    d.update(kw)
    return EmailDraft(**d)


def test_good_draft_passes():
    r = lint(good(), VIK, [Q_SSO, Q_HIRE])
    assert r.ok, r.problems


@pytest.mark.parametrize("change,problem", [
    (dict(body=GOOD_BODY.replace("Hi Vik", "Hi there")), "first name"),
    (dict(hook_quote="Acme raised a huge round last week from investors"), "hook_quote"),
    (dict(subject="A very long subject line that keeps going on"), "subject"),
    (dict(body=GOOD_BODY + " I hope this finds you well."), "banned"),
    (dict(body=GOOD_BODY.replace("Saw", "Wow!")), "exclamation"),
    (dict(body=GOOD_BODY + " See https://sprinto.com"), "URL"),
    (dict(body=GOOD_BODY + " Free on Tuesday at 3pm?"), "days, times"),
    (dict(body=GOOD_BODY.replace("Acme's", "[Company]'s")), "placeholder"),
    (dict(body=GOOD_BODY + " Customers save 65% of audit time."), "number '65%'"),
    (dict(capability_id="ai_magic"), "capability_id"),
    (dict(body=GOOD_BODY.replace(CAP, "Sprinto is great.")), "facts-file wording"),
    (dict(cta="Can we hop on a call?"), "CTA"),
    (dict(body=" ".join(["word"] * 130).replace("word", "Hi Vik,", 1)), "max 120"),
])
def test_lint_catches_each_rule(change, problem):
    r = lint(good(**change), VIK, [Q_SSO, Q_HIRE])
    assert not r.ok and any(problem.lower() in p.lower() for p in r.problems), r.problems


def test_pain_point_with_unverified_quote_is_rejected_then_regenerated():
    bad = {"pain_points": [{"pain": "a", "why_this_person": "b", "evidence_quote": "They just raised 50 million dollars"},
                           {"pain": "c", "why_this_person": "d", "evidence_quote": Q_HIRE}]}
    ok = {"pain_points": [{"pain": "a", "why_this_person": "b", "evidence_quote": Q_SSO},
                          {"pain": "c", "why_this_person": "d", "evidence_quote": Q_HIRE}]}
    fake = FakeProvider([bad, ok])
    pains, flags = pain_points(LLM(fake), FACTS, VIK, "acme.io", V)
    assert len(pains) == 2 and not flags and "REJECTED" in fake.prompts[1][1]


def test_pain_points_still_unverified_after_one_retry_are_flagged():
    bad = {"pain_points": [{"pain": "a", "why_this_person": "b", "evidence_quote": "invented sentence about the company"}] * 2}
    pains, flags = pain_points(LLM(FakeProvider([bad, bad])), FACTS, VIK, "acme.io", V)
    assert pains == [] and "pain_points_unverified" in flags


def test_failing_draft_redrafted_once_then_needs_human_edit():
    bad = {"subject": "Hi!", "body": "Hello there!", "hook_quote": "x", "capability_id": "x", "cta": "x"}
    fake = FakeProvider([bad, bad])
    d, r, prompt, flags = draft_email(LLM(fake), FACTS, VIK, [], "acme.io")
    assert not r.ok and "needs_human_edit" in flags and len(fake.prompts) == 2
    assert "FAILED THESE CHECKS" in fake.prompts[1][1]


def test_draft_redraft_fixes_it():
    bad = {"subject": "Hi!", "body": "Hello there!", "hook_quote": "x", "capability_id": "x", "cta": "x"}
    fixed = good().model_dump()
    d, r, _, flags = draft_email(LLM(FakeProvider([bad, fixed])), FACTS, VIK, [], "acme.io")
    assert r.ok and not flags


# ---------------- send_email guard ----------------

class FakeGmail:
    def __init__(self):
        self.sent = []

    def users(self):
        return self

    def messages(self):
        return self

    def send(self, userId, body):
        self.sent.append(body)
        return self

    def execute(self):
        return {"id": "m1", "threadId": "t1"}


def test_send_email_only_to_my_email(monkeypatch):
    from agent.tools import send_email as se
    g = FakeGmail()
    out = se.send_email("cold", "Subject", "Body to vik@acme.io", service=g)
    msg = email.message_from_bytes(base64.urlsafe_b64decode(g.sent[0]["raw"]))
    assert out.status == "sent" and msg["To"] == "me@example.com" and msg["Cc"] is None and msg["Bcc"] is None


def test_send_email_refuses_without_my_email(monkeypatch):
    from agent import config
    from agent.guards import GuardViolation
    from agent.tools import send_email as se
    monkeypatch.setenv("MY_EMAIL", "")
    config.settings.cache_clear()
    try:
        with pytest.raises(GuardViolation):
            se.send_email("cold", "s", "b", service=FakeGmail())
    finally:
        monkeypatch.setenv("MY_EMAIL", "me@example.com")
        config.settings.cache_clear()


def test_tool_has_no_way_to_set_a_recipient():
    from agent.tools.send_email import TOOL
    assert not {"to", "cc", "bcc", "recipient", "recipients"} & set(TOOL.input_model.model_fields)
