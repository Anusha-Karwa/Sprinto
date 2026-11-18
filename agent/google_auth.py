"""Google OAuth for the BOT account (Gmail + Calendar). One consent covers every phase.

This runs in a cloud workspace without a browser, so the flow is copy-paste:
  1. `python -m agent.google_auth` prints a Google sign-in link.
  2. Sign in AS THE BOT ACCOUNT and allow access.
  3. Google redirects to http://localhost:8765/?code=... — the page won't load (that's expected).
     Copy the FULL address from the browser bar and paste it back.
  4. token.json is saved (git-ignored) and refreshed automatically afterwards.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from agent.config import ROOT, settings

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]
REDIRECT = "http://localhost:8765/"


def _paths() -> tuple[Path, Path]:
    s = settings()
    return ROOT / s.google_credentials_file, ROOT / s.google_token_file


def credentials():
    """Load + refresh credentials. Local: token.json. Vercel: GOOGLE_TOKEN_JSON env var (the same JSON).
    Raises a clear error if setup hasn't been done."""
    import json
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    _, token = _paths()
    env_token = os.environ.get("GOOGLE_TOKEN_JSON")
    if env_token:
        creds = Credentials.from_authorized_user_info(json.loads(env_token), SCOPES)
    elif token.exists():
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    else:
        raise RuntimeError("Google is not connected yet: run `python -m agent.google_auth` (see README).")
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())          # in memory on Vercel; saved back to token.json locally
            if not env_token:
                token.write_text(creds.to_json())
        else:
            raise RuntimeError("Google token is invalid — run `python -m agent.google_auth` again.")
    return creds


def gmail():
    from googleapiclient.discovery import build
    return build("gmail", "v1", credentials=credentials(), cache_discovery=False)


def calendar():
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=credentials(), cache_discovery=False)


def start_flow():
    from google_auth_oauthlib.flow import Flow
    cred_file, _ = _paths()
    if not cred_file.exists():
        sys.exit(f"Missing {cred_file.name}: download the OAuth client JSON from Google Cloud (see README).")
    flow = Flow.from_client_secrets_file(str(cred_file), scopes=SCOPES, redirect_uri=REDIRECT)
    url, state = flow.authorization_url(access_type="offline", prompt="consent", include_granted_scopes="true")
    return flow, url


def finish_flow(flow, redirected_url: str) -> str:
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")   # the redirect is to plain-http localhost
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    flow.fetch_token(authorization_response=redirected_url.strip())
    _, token = _paths()
    token.write_text(flow.credentials.to_json())
    profile = gmail().users().getProfile(userId="me").execute()
    return profile["emailAddress"]


PENDING = ROOT / ".oauth_pending.json"   # git-ignored; holds the PKCE verifier between the two steps


def cli_url() -> str:
    import json
    flow, url = start_flow()
    PENDING.write_text(json.dumps({"code_verifier": flow.code_verifier}))
    PENDING.chmod(0o600)
    return url


def cli_finish(redirected_url: str) -> str:
    import json
    from google_auth_oauthlib.flow import Flow
    cred_file, _ = _paths()
    flow = Flow.from_client_secrets_file(str(cred_file), scopes=SCOPES, redirect_uri=REDIRECT)
    flow.code_verifier = json.loads(PENDING.read_text())["code_verifier"]
    who = finish_flow(flow, redirected_url)
    PENDING.unlink(missing_ok=True)
    return who


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "url":
    print(cli_url())
    sys.exit(0)
if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "finish":
    print("Connected as", cli_finish(sys.argv[2]))
    sys.exit(0)
if __name__ == "__main__":
    flow, url = start_flow()
    print("\n1) Open this link, sign in as the BOT account, and click Allow:\n\n" + url)
    print("\n2) The browser will land on a page that doesn't load (localhost). Copy the FULL address and paste it here:")
    who = finish_flow(flow, input("> "))
    print(f"\nConnected as {who}. token.json saved.")
    if who.lower() != (settings().bot_email or "").lower():
        print(f"WARNING: connected account {who} is not BOT_EMAIL={settings().bot_email}")
