from __future__ import annotations

from pathlib import Path

from doc_pipeline.app import access
from doc_pipeline.app.access import client_ip, is_allowed, sign_in_providers, user_key

SHARED = {"redirect_uri": "https://app.example/oauth2callback", "cookie_secret": "s3cret"}
PROVIDER = {
    "client_id": "id",
    "client_secret": "secret",
    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
}


def test_sign_in_providers_come_from_the_auth_secrets() -> None:
    assert sign_in_providers({**SHARED, "microsoft": PROVIDER, "google": PROVIDER}) == ["google", "microsoft"]
    # One provider set directly under [auth] is used without a name.
    assert sign_in_providers({**SHARED, **PROVIDER}) == []
    assert sign_in_providers({**SHARED, **PROVIDER, "client_kwargs": {"prompt": "select_account"}}) == []


def test_missing_or_incomplete_sign_in_settings_mean_no_sign_in() -> None:
    # The app then refuses to run, so a missing setting never opens it up.
    assert sign_in_providers(None) is None
    assert sign_in_providers({}) is None
    assert sign_in_providers(SHARED) is None
    assert sign_in_providers({"cookie_secret": "s3cret", "google": PROVIDER}) is None
    assert sign_in_providers({"redirect_uri": "https://app.example/oauth2callback", "google": PROVIDER}) is None


def test_any_signed_in_account_may_use_the_app_without_an_allowlist() -> None:
    assert is_allowed({"email": "someone@example.com"})
    assert is_allowed({"email": "someone@example.com"}, {"allowed_emails": [], "allowed_domains": []})


def test_the_allowlist_needs_a_verified_email() -> None:
    access = {"allowed_emails": ["Boss@Example.com"], "allowed_domains": ["@team.example"]}

    assert is_allowed({"email": "boss@example.com", "email_verified": True}, access)
    assert is_allowed({"email": "Anyone@TEAM.example", "email_verified": True}, access)
    # Some providers let account owners set any address, so only verified emails count.
    assert not is_allowed({"email": "boss@example.com"}, access)
    assert not is_allowed({"email": "boss@example.com", "email_verified": "true"}, access)
    for email in ("other@example.com", "boss@example.com.evil", "anyone@evilteam.example", "x@sub.team.example"):
        assert not is_allowed({"email": email, "email_verified": True}, access)
    assert not is_allowed({"email_verified": True}, access)
    assert is_allowed({"email": "solo@example.com", "email_verified": True}, {"allowed_emails": "solo@example.com"})


def test_run_limits_follow_the_account_not_its_email() -> None:
    claims = {"iss": "https://accounts.google.com", "sub": "1234", "email": "a@example.com"}

    assert user_key(claims) == "https://accounts.google.com|1234"
    assert user_key({**claims, "email": "b@example.com"}) == user_key(claims)


def test_the_access_module_never_imports_streamlit() -> None:
    source = Path(access.__file__).read_text(encoding="utf-8")

    assert "import streamlit" not in source and "from streamlit" not in source


def test_the_client_ip_is_the_last_forwarded_entry_stripped() -> None:
    # The trusted proxy appends its entry, so the last one is the address it
    # saw; an earlier one can be planted by the client.
    assert client_ip({"X-Forwarded-For": "203.0.113.7, 198.51.100.9, 192.0.2.1"}) == "192.0.2.1"
    assert client_ip({"X-Forwarded-For": "192.0.2.1"}) == "192.0.2.1"
    assert client_ip({"X-Forwarded-For": "  192.0.2.1\t"}) == "192.0.2.1"


def test_the_forwarded_header_name_is_matched_case_insensitively() -> None:
    assert client_ip({"x-forwarded-for": "192.0.2.1"}) == "192.0.2.1"
    assert client_ip({"X-FORWARDED-FOR": "192.0.2.1"}) == "192.0.2.1"
    assert client_ip({" x-forwarded-for ": "192.0.2.1"}) == "192.0.2.1"


def test_a_missing_blank_or_non_string_forwarded_header_means_no_ip() -> None:
    assert client_ip({}) is None
    assert client_ip({"Host": "app.example"}) is None
    assert client_ip({"X-Forwarded-For": ""}) is None
    assert client_ip({"X-Forwarded-For": "   "}) is None
    assert client_ip({"X-Forwarded-For": ","}) is None
    assert client_ip({"X-Forwarded-For": "192.0.2.1, "}) is None
    assert client_ip({"X-Forwarded-For": 192_0_2_1}) is None
    assert client_ip({"X-Forwarded-For": None}) is None
    assert client_ip({"X-Forwarded-For": ["192.0.2.1"]}) is None


def test_an_ipv6_address_is_keyed_per_64_network() -> None:
    # One subnet shares one key: the whole /64 counts as one visitor.
    assert client_ip({"X-Forwarded-For": "2001:db8::1"}) == "2001:db8::/64"
    assert client_ip({"X-Forwarded-For": "2001:db8::dead:beef"}) == "2001:db8::/64"
    assert client_ip({"X-Forwarded-For": "2001:db8:1:2::1"}) == "2001:db8:1:2::/64"


def test_an_invalid_forwarded_entry_means_no_ip() -> None:
    # The value is parsed, not trusted: garbage names no address at all.
    assert client_ip({"X-Forwarded-For": "not-an-address"}) is None
    assert client_ip({"X-Forwarded-For": "192.0.2.1, junk"}) is None
    assert client_ip({"X-Forwarded-For": "999.999.999.999"}) is None
    assert client_ip({"X-Forwarded-For": "1.2.3.4/32"}) is None


def test_an_ipv4_mapped_ipv6_address_is_keyed_as_the_ipv4_address() -> None:
    # A proxy that rewrites the address into ::ffff:a.b.c.d must not give the
    # visit a second key (or drop it into ::/64 with the whole internet).
    assert client_ip({"X-Forwarded-For": "::ffff:1.2.3.4"}) == "1.2.3.4"
    assert client_ip({"X-Forwarded-For": "::ffff:192.0.2.1"}) == "192.0.2.1"
    assert client_ip({"X-Forwarded-For": "::ffff:0102:0304"}) == "1.2.3.4"
    assert client_ip({"X-Forwarded-For": "::ffff:1.2.3.4"}) == client_ip({"X-Forwarded-For": "1.2.3.4"})
    assert client_ip({"X-Forwarded-For": "::ffff:1.2.3.5"}) != client_ip({"X-Forwarded-For": "::ffff:1.2.3.4"})
    assert client_ip({"X-Forwarded-For": "::ffff:1.2.3.4"}) != "::/64"


def test_no_forwarded_value_ever_raises() -> None:
    # Whatever a proxy or a client sends, the worst outcome is "no usable IP".
    for value in (
        "fe80::1%eth0",  # a scope id: keyed by its scope-free /64, never an exception
        "::ffff:1.2.3.4%eth0",
        "1.2.3.4 ",
        "\n192.0.2.1 DROP",
        "1.2.3.4,,5.6.7.8",
        "gopher://192.0.2.1",
        "​",
        "x" * 500,
    ):
        parsed = client_ip({"X-Forwarded-For": value})
        assert parsed is None or (parsed and "%" not in parsed)
