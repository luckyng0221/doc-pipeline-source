"""Who may use the upload app: sign-in providers and the allowlist (no Streamlit here)."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any

# Keys under [auth] that hold provider options rather than a named provider.
NOT_PROVIDERS = frozenset({"client_kwargs"})


def sign_in_providers(auth: Mapping[str, Any] | None) -> list[str] | None:
    """Provider names for st.login() from the [auth] secrets section.

    Named providers ([auth.google], [auth.microsoft]) come back sorted. [] means
    one provider set directly under [auth], which st.login() takes without a
    name. None means sign-in is not set up, or lacks redirect_uri or
    cookie_secret.
    """
    if not auth or not auth.get("redirect_uri") or not auth.get("cookie_secret"):
        return None
    named = sorted(name for name, value in auth.items() if isinstance(value, Mapping) and name not in NOT_PROVIDERS)
    if named:
        return named
    return [] if auth.get("client_id") else None


def is_allowed(claims: Mapping[str, Any], access: Mapping[str, Any] | None = None) -> bool:
    """Whether a signed-in account may use the app, per the [access] secrets section.

    Without allowed_emails or allowed_domains, any signed-in account may.
    Otherwise the provider must vouch for the email (email_verified), since
    some providers let account owners set any address, and the email must be
    listed or belong to a listed domain.
    """
    access = access or {}
    emails = _lowered(access.get("allowed_emails"))
    domains = _lowered(access.get("allowed_domains"), strip="@")
    if not emails and not domains:
        return True
    email = str(claims.get("email") or "").strip().casefold()
    if not email or claims.get("email_verified") is not True:
        return False
    return email in emails or email.rpartition("@")[2] in domains


def user_key(claims: Mapping[str, Any]) -> str:
    """A stable id for run limits: issuer and subject, which unlike an email never change."""
    return f"{claims.get('iss', '')}|{claims.get('sub', '')}"


def client_ip(headers: Mapping[str, Any]) -> str | None:
    """The address an anonymous visitor's run limits count per, or None for no usable one.

    The trusted proxy appends its own entry to the X-Forwarded-For list, so the
    last one is the address it saw; an earlier one can be planted by the client
    to reset its limits at will. That entry is parsed with the stdlib
    ``ipaddress`` module: a valid IPv4 address keys as itself, an IPv6 address
    as its /64 network (so one subnet shares one key), and an IPv4-mapped IPv6
    address (``::ffff:a.b.c.d``) as the plain IPv4 address. None when the
    header is missing, not a string, or holds no valid address - the caller's
    no-usable-IP path.
    """
    for name, value in headers.items():
        if str(name).strip().casefold() != "x-forwarded-for":
            continue
        if not isinstance(value, str):
            return None
        address = value.rsplit(",", 1)[-1].strip()
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            return None
        if isinstance(parsed, ipaddress.IPv6Address):
            mapped = parsed.ipv4_mapped
            if mapped is not None:
                return str(mapped)
            # The /64 step stays inside the try: a scope id parses as an
            # address but only a scope-free network is a usable key.
            return str(ipaddress.ip_network(f"{parsed}/64", strict=False))
        return str(parsed)
    return None


def _lowered(values: Any, strip: str = "") -> frozenset[str]:
    if isinstance(values, str):
        values = [values]
    cleaned = (str(value).strip().lstrip(strip).casefold() for value in values or ())
    return frozenset(value for value in cleaned if value)
