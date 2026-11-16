"""Fake HTTP for scraper tests — no network."""
import json


class FakeResponse:
    def __init__(self, url, status=200, text="", headers=None, final_url=None):
        self.url = final_url or url
        self.status_code = status
        self.text = text
        self.headers = {"content-type": "text/html; charset=utf-8", **(headers or {})}

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self, routes):
        self.routes = routes          # url → FakeResponse | Exception
        self.calls = []
        self.headers = {}

    def get(self, url, **kw):
        self.calls.append(url)
        r = self.routes.get(url)
        if r is None:
            return FakeResponse(url, 404, "not found")
        if isinstance(r, Exception):
            raise r
        return r


def page(body, title="Acme", extra_head=""):
    return f"<html><head><title>{title}</title>{extra_head}</head><body>{body}</body></html>"


LONG = " ".join(["Acme helps finance teams at fast growing businesses close their books faster every month."] * 30)
