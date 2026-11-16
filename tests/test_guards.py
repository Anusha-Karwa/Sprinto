import pytest

from agent.guards import GuardViolation, assert_recipients


def test_only_my_email_passes():
    assert_recipients(["me@example.com"])
    assert_recipients(["ME@Example.com "], attendees=["me@example.com"])


@pytest.mark.parametrize("field", ["to", "cc", "bcc", "attendees"])
def test_prospect_address_never_reaches_any_field(field):
    kw = {"to": ["me@example.com"], field: ["cto@prospect.io"]}
    with pytest.raises(GuardViolation):
        assert_recipients(**kw)


def test_empty_recipients_refused():
    with pytest.raises(GuardViolation):
        assert_recipients([])
