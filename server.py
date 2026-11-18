"""Vercel entrypoint: Vercel looks for a top-level `app` in server.py. The real app lives in web/main.py."""
from web.main import app  # noqa: F401
