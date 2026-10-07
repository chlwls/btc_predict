"""Minimal Upbit REST client (KRW spot market orders only)."""

import hashlib
import uuid
from urllib.parse import urlencode, unquote

import jwt
import requests

BASE = "https://api.upbit.com/v1"


class UpbitClient:
    def __init__(self, access_key, secret_key, timeout=10):
        self.ak, self.sk, self.timeout = access_key, secret_key, timeout

    # ---- auth
    def _headers(self, params=None):
        payload = {"access_key": self.ak, "nonce": str(uuid.uuid4())}
        if params:
            q = unquote(urlencode(params, doseq=True)).encode()
            payload["query_hash"] = hashlib.sha512(q).hexdigest()
            payload["query_hash_alg"] = "SHA512"
        token = jwt.encode(payload, self.sk, algorithm="HS256")
        return {"Authorization": f"Bearer {token}"}

    def _check(self, r):
        if r.status_code >= 400:
            raise RuntimeError(f"Upbit {r.status_code}: {r.text}")
        return r.json()

    # ---- public
    def krw_markets(self):
        r = requests.get(f"{BASE}/market/all", timeout=self.timeout)
        return {m["market"] for m in self._check(r) if m["market"].startswith("KRW-")}

    def prices(self, markets):
        if not markets:
            return {}
        r = requests.get(f"{BASE}/ticker", params={"markets": ",".join(markets)}, timeout=self.timeout)
        return {t["market"]: float(t["trade_price"]) for t in self._check(r)}

    # ---- private
    def balances(self):
        r = requests.get(f"{BASE}/accounts", headers=self._headers(), timeout=self.timeout)
        return {a["currency"]: float(a["balance"]) for a in self._check(r)}

    def buy_market_krw(self, market, krw):
        params = {"market": market, "side": "bid", "ord_type": "price", "price": str(int(krw))}
        r = requests.post(f"{BASE}/orders", json=params, headers=self._headers(params), timeout=self.timeout)
        return self._check(r)

    def sell_market(self, market, volume):
        params = {"market": market, "side": "ask", "ord_type": "market", "volume": f"{volume:.8f}"}
        r = requests.post(f"{BASE}/orders", json=params, headers=self._headers(params), timeout=self.timeout)
        return self._check(r)


class DryRunClient:
    """Reads real public data (and real balances if keys are given) but never orders."""

    def __init__(self, real=None, virtual_krw=500_000):
        self.real = real
        self.virtual_krw = virtual_krw
        self._pub = UpbitClient("", "")

    def krw_markets(self):
        return self._pub.krw_markets()

    def prices(self, markets):
        return self._pub.prices(markets)

    def balances(self):
        return self.real.balances() if self.real else {"KRW": float(self.virtual_krw)}

    def buy_market_krw(self, market, krw):
        return {"dry_run": True, "market": market, "side": "bid", "krw": krw}

    def sell_market(self, market, volume):
        return {"dry_run": True, "market": market, "side": "ask", "volume": volume}
