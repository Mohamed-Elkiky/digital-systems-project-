"""Local-file user authentication for the dashboard.

Users are stored in data/users.json with PBKDF2-hashed passwords.
API keys are stored per-user and never logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
from pathlib import Path
from typing import Any

import certifi
import hmac as _hmac
import time
import urllib.error
import urllib.request

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_FILE = PROJECT_ROOT / "data" / "users.json"

BYBIT_DEMO_BASE = "https://api-demo.bybit.com"
BYBIT_TESTNET_BASE = "https://api-testnet.bybit.com"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> dict[str, Any]:
    if not USERS_FILE.exists():
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)


def _save(data: dict[str, Any]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()


# ---------------------------------------------------------------------------
# Public auth API
# ---------------------------------------------------------------------------

def register(
    username: str,
    password: str,
    api_key: str = "",
    api_secret: str = "",
    use_demo: bool = True,
) -> tuple[bool, str]:
    """Register a new user.  Returns (success, message)."""
    username = username.strip().lower()
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username may only contain letters, numbers, - and _."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load()
    if username in users:
        return False, "Username already taken."

    salt = os.urandom(16).hex()
    users[username] = {
        "password_hash": _hash(password, salt),
        "salt": salt,
        "bybit_api_key": api_key.strip(),
        "bybit_api_secret": api_secret.strip(),
        "bybit_use_demo": use_demo,
    }
    _save(users)
    return True, "Account created successfully."


def login(username: str, password: str) -> tuple[bool, str]:
    """Verify credentials.  Returns (success, message)."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Enter username and password."
    users = _load()
    if username not in users:
        return False, "Invalid username or password."
    u = users[username]
    if _hash(password, u["salt"]) != u["password_hash"]:
        return False, "Invalid username or password."
    return True, "Logged in."


def get_user(username: str) -> dict[str, Any]:
    """Return stored data for *username* (empty dict if not found)."""
    return _load().get(username.lower(), {})


def update_api_keys(
    username: str,
    api_key: str,
    api_secret: str,
    use_demo: bool = True,
) -> None:
    """Persist Bybit API credentials for *username*."""
    users = _load()
    key = username.lower()
    if key in users:
        users[key]["bybit_api_key"] = api_key.strip()
        users[key]["bybit_api_secret"] = api_secret.strip()
        users[key]["bybit_use_demo"] = use_demo
        _save(users)


def update_password(
    username: str, old_password: str, new_password: str
) -> tuple[bool, str]:
    """Change password after verifying the old one."""
    ok, msg = login(username, old_password)
    if not ok:
        return False, "Current password is incorrect."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    users = _load()
    u = users[username.lower()]
    salt = os.urandom(16).hex()
    u["password_hash"] = _hash(new_password, salt)
    u["salt"] = salt
    _save(users)
    return True, "Password updated successfully."


# ---------------------------------------------------------------------------
# Bybit Demo API helpers
# ---------------------------------------------------------------------------

def _bybit_headers(api_key: str, api_secret: str, params_str: str) -> dict[str, str]:
    ts = str(int(time.time() * 1000))
    recv = "5000"
    sign_str = ts + api_key + recv + params_str
    sig = _hmac.new(
        api_secret.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return {
        "X-BAPI-API-KEY": api_key,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-SIGN": sig,
        "X-BAPI-RECV-WINDOW": recv,
        "Content-Type": "application/json",
    }


def test_bybit_connection(
    api_key: str, api_secret: str, use_demo: bool = True
) -> tuple[bool, str]:
    """Test connectivity and authentication against the Bybit demo/testnet API.

    Returns (success, message).
    """
    base = BYBIT_DEMO_BASE if use_demo else BYBIT_TESTNET_BASE

    # Step 1 — server reachability (no auth required)
    try:
        with urllib.request.urlopen(f"{base}/v5/market/time", timeout=6, context=_SSL_CTX) as r:
            data = json.loads(r.read())
        if data.get("retCode") != 0:
            return False, "Bybit server responded but returned an error."
    except urllib.error.URLError as e:
        return False, f"Cannot reach Bybit server: {e.reason}"
    except Exception as e:
        return False, f"Connectivity error: {e}"

    if not api_key or not api_secret:
        return False, "Server reachable, but no API credentials entered yet."

    # Step 2 — authenticated endpoint
    try:
        params = "accountType=UNIFIED"
        headers = _bybit_headers(api_key, api_secret, params)
        url = f"{base}/v5/account/wallet-balance?{params}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6, context=_SSL_CTX) as r:
            data = json.loads(r.read())
        if data.get("retCode") == 0:
            return True, "Authentication successful — Bybit demo account connected."
        return False, f"Auth failed: {data.get('retMsg', 'Unknown error')}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, f"Error: {e}"


def place_order(
    api_key: str,
    api_secret: str,
    use_demo: bool,
    mode: str,
    symbol: str = "BTCUSDT",
    side: str = "Buy",
    qty: str = "0.001",
) -> tuple[bool, str, dict]:
    """Place a market order on Bybit demo/testnet.

    *mode* is ``"Spot"`` or ``"Futures"`` (case-insensitive).
    Returns ``(success, message, raw_response)``.
    """
    base = BYBIT_DEMO_BASE if use_demo else BYBIT_TESTNET_BASE
    category = "spot" if mode.lower() == "spot" else "linear"

    body: dict[str, Any] = {
        "category": category,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": qty,
    }
    if category == "linear":
        body["positionIdx"] = 0  # one-way mode

    body_str = json.dumps(body, separators=(",", ":"))
    headers = _bybit_headers(api_key, api_secret, body_str)

    try:
        req = urllib.request.Request(
            f"{base}/v5/order/create",
            data=body_str.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as r:
            data = json.loads(r.read())
        if data.get("retCode") == 0:
            order_id = data.get("result", {}).get("orderId", "")
            return True, f"Order placed — ID: {order_id}", data
        return False, f"Order rejected: {data.get('retMsg', 'Unknown')}", data
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode()
        except Exception:
            err_body = str(e.reason)
        return False, f"HTTP {e.code}: {err_body}", {}
    except Exception as e:
        return False, f"Error placing order: {e}", {}
