"""Content-free audit logging for the upload app (no Streamlit here).

One line per event goes to the dedicated ``doc_pipeline.audit`` logger: a run
started or ended, a run refused by the limits, an access refusal, a sign-out
that stopped a run. The lines let the owner spot abuse in the server's logs
without the log ever holding content: every value must come from a fixed set
of codes, or be a plain count, and anything else drops the whole line, so an
email, a file name, or an API key cannot reach the log even when a caller
passes one.

People are identified only by ``pseudonym()``: the first 12 hex characters of
HMAC-SHA256 of the stripped, lowercased email keyed with the ``[auth]
cookie_secret`` (rotating that secret changes every id at once).
``anonymous`` covers anonymous use; a missing key or email gives ``unknown``.

The logger carries its own stderr handler because ``streamlit.logger``
configures only the loggers created through it: the root logger keeps its
WARNING default and has no handler, so an INFO record on this logger would be
dropped. The handler is attached once per process - a lock, plus a mark on
the handler itself, so even a re-import of this module cannot add a second
one - and ``propagate`` stays True, so pytest's caplog and any handler a host
puts on the root logger still see every record.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import sys
import threading

from .settings import PROVIDER_LABELS

LOGGER_NAME = "doc_pipeline.audit"
# Streamlit's own console format, so the lines look at home in the app logs.
_FORMAT = "%(asctime)s %(levelname) -7s %(name)s: %(message)s"

# The fixed vocabularies an audit line may draw on. The event name and the
# user id are checked whole; a field also accepts several of its codes,
# comma-joined, where one line can carry more than one (refusal reasons,
# model providers).
EVENTS = frozenset({"run_started", "run_ended", "run_refused", "access_refused", "signout_stopped_run"})
REASONS = frozenset(
    {
        "user_active",
        "server_busy",
        "too_many_files",
        "file_too_large",
        "hour_quota",
        "day_quota",
        "no_client_ip",
        "test_quota",
        "allowlist",
        "anonymous_secret",
        "no_auth",
    }
)
OUTCOMES = frozenset({"finished", "cancelled", "timed_out", "failed", "signout"})
MODES = frozenset({"models", "rules"})
PROVIDERS = frozenset(PROVIDER_LABELS)
# Field name -> (its codes, whether a comma-joined list of them is allowed).
_CODES = {
    "reason": (REASONS, True),
    "outcome": (OUTCOMES, False),
    "mode": (MODES, False),
    "providers": (PROVIDERS, True),
}
# Event and field names are ours: lower-case words with underscores.
_NAME = re.compile(r"[a-z][a-z0-9_]*")
# The user id: a pseudonym, or one of the two fixed ids.
_USER = re.compile(r"[0-9a-f]{12}|anonymous|unknown")
# clean_value's allowlist: strings of these characters, 1 to 64 of them.
_VALUE = re.compile(r"[A-Za-z0-9_.,:+/@-]{1,64}")

_LOCK = threading.Lock()
# Marks the stderr handler this module attached. A module flag alone would not
# survive a re-import, which would then add a second handler; the mark sits on
# the handler itself, and loggers are singletons, so it is always found.
_HANDLED = "doc_pipeline_audit_handler"


def _logger() -> logging.Logger:
    """The audit logger: at INFO, with exactly one stderr handler per process."""
    with _LOCK:
        logger = logging.getLogger(LOGGER_NAME)
        # Set on every call: the handler is attached once, but a host (or a
        # test's log capture) may have moved the level since.
        logger.setLevel(logging.INFO)
        if not any(getattr(handler, _HANDLED, False) for handler in logger.handlers):
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_FORMAT))
            setattr(handler, _HANDLED, True)
            logger.addHandler(handler)
        # propagate stays True (the logging default): pytest's caplog and any
        # root handler a host adds must still see the records.
        return logger


def pseudonym(email: str, key: str | None) -> str:
    """A stable id for the log that never carries the email itself.

    The first 12 hex characters of HMAC-SHA256 of the stripped, lowercased
    email keyed with the app's ``[auth] cookie_secret`` (the README has the
    one-line command to recompute one). ``unknown`` without a key or without
    an email: those visitors cannot be told apart, so the log says nothing.
    """
    if not key or not email.strip():
        return "unknown"
    return hmac.new(key.encode(), email.strip().lower().encode(), hashlib.sha256).hexdigest()[:12]


def clean_value(value: object) -> str:
    """One value as log text, or ValueError for anything that could forge a line.

    Integers (not bools) and strings of at most 64 characters drawn from a
    fixed set: no spaces, no ``=``, no newlines, so a value can neither add a
    field nor a line. ``log_event`` is stricter still (its string fields must
    be known codes); a refused value is described, never quoted.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return int.__repr__(value)  # not str(): a subclass could override __str__
    if isinstance(value, str) and _VALUE.fullmatch(value):
        return value
    raise ValueError(f"refused audit value of type {type(value).__name__}")


def log_event(event: str, *, user: str = "unknown", warning: bool = False, **fields: object) -> None:
    """Emit one ``event=<code> user=<id> k=v …`` line, in insertion order; never raises.

    Every name and value is checked against a fixed set - an event name, a
    user id, a field's reason, outcome, mode, or provider codes, or an
    integer - and one value outside its set drops the whole line without
    raising, so neither a bug nor a hostile caller can put content or a
    secret into the log, and logging can never break a run or the page.
    """
    try:
        parts = [f"event={_code(event, EVENTS)}", f"user={_user(user)}"]
        parts += [f"{name}={_field(name, value)}" for name, value in fields.items()]
        line = " ".join(parts)
        logger = _logger()
        if warning:
            logger.warning(line)
        else:
            logger.info(line)
    except Exception:
        pass  # a lost audit line must never break a run or the page


def _code(value: str, codes: frozenset[str]) -> str:
    if value not in codes:
        raise ValueError("value is not one of the audit codes")
    return value


def _user(user: str) -> str:
    if not isinstance(user, str) or not _USER.fullmatch(user):
        raise ValueError("refused audit user id")
    return user


def _field(name: str, value: object) -> str:
    """One field's text: an integer count, or codes from that field's own set."""
    if not _NAME.fullmatch(name):
        raise ValueError("refused audit field name")
    if isinstance(value, int) and not isinstance(value, bool):
        return int.__repr__(value)  # not str(): a subclass could override __str__
    allowed = _CODES.get(name)
    if allowed is None or not isinstance(value, str):
        raise ValueError(f"field {name}: value is not one of its codes")
    codes, joined = allowed
    parts = value.split(",") if joined else [value]
    if value and all(part in codes for part in parts):
        return value
    raise ValueError(f"field {name}: value is not one of its codes")
