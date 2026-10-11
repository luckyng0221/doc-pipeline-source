from __future__ import annotations

import hashlib
import hmac
import logging
import re
import threading
from concurrent.futures import Future
from pathlib import Path

import pymupdf
import pytest

from doc_pipeline.app import audit
from doc_pipeline.app.session import RunControl, RunQuota, RunResult, RunSlots, Upload, upload_refusal_codes
from doc_pipeline.app.settings import AppLimits
from doc_pipeline.errors import RunStopped
from test_app import AUTH

ROOT = Path(__file__).parents[1]
# The cookie secret the sample sign-in setup (test_app.AUTH) is keyed with.
SECRET = AUTH["cookie_secret"]


def _app(tmp_path: Path, monkeypatch, *, anonymous: bool = True):
    """The upload page with its folders in tmp_path; anonymous skips sign-in, as for local use."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DOC_PIPELINE_WORKSPACES", str(tmp_path))
    if anonymous:
        monkeypatch.setenv("DOC_PIPELINE_ALLOW_ANONYMOUS", "1")
    else:
        monkeypatch.delenv("DOC_PIPELINE_ALLOW_ANONYMOUS", raising=False)
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)


@pytest.fixture(autouse=True)
def _no_real_process_hardening(monkeypatch):
    """AppTest runs the page inside this pytest process: never really prctl it.

    The once-per-process globals go back to their "not yet done" value so the
    page's initializers really run (test_hardening.py covers the real call).
    """
    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "set_non_dumpable", lambda: True)
    monkeypatch.setattr(ui, "import_provider_sdks", lambda: None)
    monkeypatch.setattr(ui, "_hardened", None)
    monkeypatch.setattr(ui, "_sdks_imported", None)
    monkeypatch.setattr(ui, "_anonymous", None)
    monkeypatch.setattr(ui, "_anonymous_refused", None)
    monkeypatch.setattr(ui, "_slots", None)
    monkeypatch.setattr(ui, "_quota", None)
    monkeypatch.setattr(ui, "_test_quota", None)
    monkeypatch.setattr(ui, "_allowlist_warned", None)
    yield


class _SignedIn(dict):
    """Stands in for st.user after a sign-in."""

    def __init__(self, **claims: object) -> None:
        super().__init__(is_logged_in=True, iss="https://accounts.google.com", sub="42", **claims)

    def to_dict(self) -> dict:
        return dict(self)


def _pdf(pages: list[str]) -> bytes:
    document = pymupdf.open()
    for text in pages:
        document.new_page().insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


class _PageCrashesMidRun:
    """Progress stand-in: the page breaks on its second poll, so the job stays
    shown mid-run with its Sign out button, the way a live rerun would."""

    def __init__(self) -> None:
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return (0.0 if self.calls == 1 else 2.0, "Working…", "")  # 2.0 breaks st.progress


def _audit(caplog) -> list[tuple[logging.LogRecord, dict[str, str]]]:
    """The audit records with their `key=value` fields, in order.

    The split on spaces only works because the values are all fixed codes and
    counts: that is the content-free promise, asserted on every line below.
    """
    return [
        (record, dict(part.split("=", 1) for part in record.getMessage().split(" ")))
        for record in caplog.records
        if record.name == audit.LOGGER_NAME
    ]


def test_the_audit_module_never_imports_streamlit() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")

    assert "import streamlit" not in source and "from streamlit" not in source


def test_each_event_logs_one_line_with_its_fields_at_its_level(caplog) -> None:
    user = "0123456789ab"

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit.log_event("run_started", user=user, files=2, bytes=2048, mode="models", providers="gemini,openai")
        audit.log_event(
            "run_ended", user=user, outcome="finished", seconds=95, files_done=2, files_failed=0, documents=3
        )
        audit.log_event("run_refused", user=user, warning=True, reason="too_many_files,file_too_large")
        audit.log_event("access_refused", user=audit.pseudonym("Ana@example.com ", "k"), warning=True, reason="allowlist")
        audit.log_event("signout_stopped_run", user="anonymous")

    records = _audit(caplog)
    assert [record.levelno for record, _ in records] == [
        logging.INFO,
        logging.INFO,
        logging.WARNING,
        logging.WARNING,
        logging.INFO,
    ]
    assert [fields for _, fields in records] == [
        {
            "event": "run_started",
            "user": user,
            "files": "2",
            "bytes": "2048",
            "mode": "models",
            "providers": "gemini,openai",
        },
        {
            "event": "run_ended",
            "user": user,
            "outcome": "finished",
            "seconds": "95",
            "files_done": "2",
            "files_failed": "0",
            "documents": "3",
        },
        {"event": "run_refused", "user": user, "reason": "too_many_files,file_too_large"},
        {"event": "access_refused", "user": audit.pseudonym("Ana@example.com ", "k"), "reason": "allowlist"},
        {"event": "signout_stopped_run", "user": "anonymous"},
    ]


def test_the_pseudonym_follows_the_documented_recipe() -> None:
    # The recipe as documented: HMAC-SHA256 of email.strip().lower(), keyed
    # with the cookie secret, first 12 hex characters.
    for email in ("Boss@Example.com ", "\tboss@example.com\n", "BOSS@example.com"):
        expected = hmac.new(SECRET.encode(), email.strip().lower().encode(), hashlib.sha256).hexdigest()[:12]
        assert audit.pseudonym(email, SECRET) == expected

    # lower(), not casefold(): a sharp s must survive unchanged.
    lowered = hmac.new(b"k", "straße@example.com".encode(), hashlib.sha256).hexdigest()[:12]
    casefolded = hmac.new(b"k", "straße@example.com".casefold().encode(), hashlib.sha256).hexdigest()[:12]
    assert lowered != casefolded
    assert audit.pseudonym("Straße@example.com", "k") == lowered

    assert re.fullmatch(r"[0-9a-f]{12}", audit.pseudonym("boss@example.com", SECRET))
    assert audit.pseudonym("boss@example.com", "") == "unknown"
    assert audit.pseudonym("boss@example.com", None) == "unknown"
    assert audit.pseudonym("", SECRET) == "unknown"
    assert audit.pseudonym("   ", SECRET) == "unknown"


def test_clean_value_takes_integers_and_simple_codes_only() -> None:
    assert audit.clean_value(0) == "0"
    assert audit.clean_value(-3) == "-3"
    assert audit.clean_value(2048) == "2048"
    assert audit.clean_value("server_busy") == "server_busy"
    # clean_value is only the character check: an email or a file name passes
    # it, which is why log_event's per-field code sets are stricter.
    assert audit.clean_value("boss@example.com") == "boss@example.com"
    assert audit.clean_value("batch.pdf") == "batch.pdf"
    for bad in (
        "two words",
        "a=b",
        "line\nbreak",
        "tab\tstop",
        "",
        "x" * 65,
        1.5,
        True,
        False,
        None,
        ["a"],
        ("a",),
    ):
        with pytest.raises(ValueError):
            audit.clean_value(bad)


def test_integers_are_written_as_their_digits_whatever_str_says(caplog) -> None:
    class Sneaky(int):
        def __str__(self) -> str:
            return "x=y"

        __repr__ = __str__

    assert audit.clean_value(Sneaky(7)) == "7"
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit.log_event("run_started", user="0123456789ab", files=Sneaky(3))

    assert [fields for _, fields in _audit(caplog)] == [{"event": "run_started", "user": "0123456789ab", "files": "3"}]


def test_emails_file_names_and_keys_drop_the_whole_line(caplog) -> None:
    key = "sk-" + "a" * 30

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        for value in ("boss@example.com", "batch.pdf", key):
            audit.log_event("run_started", user=value, files=1)
            audit.log_event("run_started", user="0123456789ab", reason=value)
            audit.log_event("run_started", user="0123456789ab", outcome=value)
            audit.log_event("run_started", user="0123456789ab", mode=value)
            audit.log_event("run_started", user="0123456789ab", providers=value)
            audit.log_event("run_started", user="0123456789ab", note=value)  # not a known field
            audit.log_event(value)  # not a known event
            audit.log_event("run_ended", user="0123456789ab", outcome="finished", seconds=value)

    assert _audit(caplog) == []


def test_counts_must_be_whole_integers(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        for value in (True, False, 1.5, None):
            audit.log_event("run_started", user="0123456789ab", files=value)

    assert _audit(caplog) == []


def test_setup_is_idempotent_and_keeps_propagation_on() -> None:
    logger = audit._logger()
    logger.setLevel(logging.WARNING)  # a host moved the level; setup puts it back

    for _ in range(3):
        assert audit._logger() is logger

    assert logger.level == logging.INFO
    assert logger.propagate is True
    marked = [handler for handler in logger.handlers if getattr(handler, audit._HANDLED, False)]
    assert len(marked) == 1
    assert isinstance(marked[0], logging.StreamHandler)


def test_a_fresh_module_state_still_finds_its_handler(monkeypatch) -> None:
    # A re-import starts from fresh module state; the mark on the handler is
    # what keeps a second one off the logger, which is a singleton.
    logger = audit._logger()
    handlers = list(logger.handlers)
    monkeypatch.setattr(audit, "_LOCK", threading.Lock())

    assert audit._logger() is logger
    assert logger.handlers == handlers


def test_lines_reach_stderr_at_info_with_a_quiet_root(capfd) -> None:
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    root.setLevel(logging.WARNING)
    root.handlers.clear()  # the Streamlit default: root at WARNING, no handlers
    logger = logging.getLogger(audit.LOGGER_NAME)
    saved_audit_handlers = list(logger.handlers)
    # This process's handler points at an earlier capture's stderr, so attach
    # it afresh, the way the first page load of a server process does.
    for handler in [item for item in logger.handlers if getattr(item, audit._HANDLED, False)]:
        logger.removeHandler(handler)
    try:
        audit.log_event("run_started", user="0123456789ab", files=1)
        lines = [line for line in capfd.readouterr().err.splitlines() if "event=run_started" in line]
        assert len(lines) == 1
        assert " INFO " in lines[0] and "user=0123456789ab" in lines[0]

        # Reruns set the logger up again; still exactly one line per event.
        audit._logger()
        audit._logger()
        audit.log_event("run_ended", user="0123456789ab", outcome="finished", seconds=2)
        lines = [line for line in capfd.readouterr().err.splitlines() if "event=run_ended" in line]
        assert len(lines) == 1
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers
        # Exactly the handlers from before the test: the one it attached goes,
        # the marked one it removed comes back.
        logger.handlers[:] = saved_audit_handlers


def test_a_raising_logger_never_breaks_log_event(monkeypatch, caplog) -> None:
    class Broken:
        def info(self, *args: object) -> None:
            raise RuntimeError("stderr is gone")

        def warning(self, *args: object) -> None:
            raise RuntimeError("stderr is gone")

    def no_logger() -> logging.Logger:
        raise OSError("no logger for you")

    monkeypatch.setattr(audit, "_logger", no_logger)
    audit.log_event("run_started", user="0123456789ab", files=1)  # must not raise
    monkeypatch.setattr(audit, "_logger", Broken)
    audit.log_event("run_refused", warning=True, reason="server_busy")  # must not raise

    assert _audit(caplog) == []


def test_quota_refusal_reasons_are_known_codes(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit.log_event("run_refused", user="anonymous", warning=True, reason="hour_quota")
        audit.log_event("run_refused", user="0123456789ab", warning=True, reason="day_quota")

    records = _audit(caplog)
    assert [fields for _, fields in records] == [
        {"event": "run_refused", "user": "anonymous", "reason": "hour_quota"},
        {"event": "run_refused", "user": "0123456789ab", "reason": "day_quota"},
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING, logging.WARNING]


def test_a_page_quota_refusal_logs_hour_quota_once_per_session(tmp_path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import ui

    quota = RunQuota(1, 40)  # the real monotonic clock: this IP's hour stays spent
    assert quota.admit("ip:192.0.2.1") is None
    monkeypatch.setattr(ui, "_quota", quota)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()
        records = _audit(caplog)
        app.button(key="start").click().run()  # pressed again: shown again, logged once

    assert not app.exception
    assert any("try again in" in warning.value for warning in app.warning)  # still shown the second time
    assert [fields for _, fields in records] == [
        {"event": "run_refused", "user": "anonymous", "reason": "hour_quota"}
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]
    assert _audit(caplog) == records
    # Content-free: the file name and its text never reach the line.
    joined = " ".join(record.getMessage() for record, _ in records)
    assert "batch.pdf" not in joined and "INVOICE" not in joined and "A-1" not in joined


def test_an_ip_keyed_run_and_its_refusals_never_log_the_address(tmp_path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import ui

    # An IPv6 visit keyed by its /64: neither the raw X-Forwarded-For value,
    # nor the addresses in it, nor the normalised ip: key they become may reach
    # an audit line - not on the run's own lines and not on a refusal's.
    forwarded = "2001:db8::1, 10.0.0.9, 2001:db8::2"
    monkeypatch.setattr(ui, "_client_ip", lambda headers: ui.client_ip({"X-Forwarded-For": forwarded}))
    quota = RunQuota(1, 40)  # the real clock: the hour stays spent after the first run
    monkeypatch.setattr(ui, "_quota", quota)
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()  # the visit's run starts and finishes
        records = _audit(caplog)
        app.button(key="start").click().run()  # the next one is refused at the hour cap

    assert not app.exception
    assert [fields["event"] for _, fields in records] == ["run_started", "run_ended"]
    assert all(fields["user"] == "anonymous" for _, fields in records)
    assert [fields["reason"] for _, fields in _audit(caplog)[len(records) :]] == ["hour_quota"]
    joined = " ".join(record.getMessage() for record, _ in _audit(caplog))
    for absent in (forwarded, "2001:db8::1", "10.0.0.9", "2001:db8::2", "2001:db8::/64", "ip:2001:db8::/64"):
        assert absent not in joined


def test_upload_refusal_codes_match_the_messages() -> None:
    limits = AppLimits(max_files=5, max_file_mb=1)
    small = Upload("a.pdf", b"x")
    big = Upload("big.pdf", b"x" * (1024 * 1024 + 1))

    assert upload_refusal_codes([], limits) == []
    assert upload_refusal_codes([small] * 5, limits) == []
    assert upload_refusal_codes([small] * 6, limits) == ["too_many_files"]
    assert upload_refusal_codes([big], limits) == ["file_too_large"]
    assert upload_refusal_codes([big] * 6, limits) == ["too_many_files", "file_too_large"]


def test_slot_refusals_keep_their_messages_and_gain_codes() -> None:
    slots = RunSlots(running=1, waiting=0)
    assert slots.admit("ana") is None

    active = slots.admit("ana")
    assert active == "You already have a run in progress. Wait for it to finish, then start the next one."
    assert active.code == "user_active"

    busy = slots.admit("ben")
    assert busy == "The server is busy with other people's runs. Please try again in a few minutes."
    assert busy.code == "server_busy"


def test_every_outcome_maps_from_the_job_state() -> None:
    from doc_pipeline.app import ui

    def result(stopped: str | None) -> RunResult:
        return RunResult([], b"", None, {}, [], stopped=stopped)

    cancelled = RunControl()
    cancelled.cancelled.set()
    signed_out = RunControl()
    signed_out.cancelled.set()
    signed_out.workspace_removed.set()
    over = RunControl()

    assert ui._outcome(None, None, RunControl()) == "cancelled"  # cancelled while queued
    assert ui._outcome(None, RuntimeError("boom"), RunControl()) == "failed"
    assert ui._outcome(None, RunStopped("the run was cancelled"), cancelled) == "cancelled"
    assert ui._outcome(None, RunStopped("the run's workspace was removed"), signed_out) == "signout"
    assert ui._outcome(None, RunStopped("the run passed its 1-minute time limit"), over) == "timed_out"
    assert ui._outcome(result(None), None, RunControl()) == "finished"
    assert ui._outcome(result("the run was cancelled"), None, cancelled) == "cancelled"
    assert ui._outcome(result("the run's workspace was removed"), None, signed_out) == "signout"
    assert ui._outcome(result("the run passed its 1-minute time limit"), None, over) == "timed_out"


def test_a_full_anonymous_run_logs_one_start_and_one_end(tmp_path, monkeypatch, caplog) -> None:
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()

    assert not app.exception
    assert "result" in app.session_state
    records = _audit(caplog)
    assert [fields["event"] for _, fields in records] == ["run_started", "run_ended"]
    assert [record.levelno for record, _ in records] == [logging.INFO, logging.INFO]
    assert records[0][1] == {
        "event": "run_started",
        "user": "anonymous",
        "files": "1",
        "bytes": str(len(data)),
        "mode": "rules",
    }
    assert records[1][1]["user"] == "anonymous"
    assert records[1][1]["outcome"] == "finished"
    assert re.fullmatch(r"\d+", records[1][1]["seconds"])
    assert set(records[1][1]) == {"event", "user", "outcome", "seconds", "files_done", "files_failed", "documents"}
    assert (records[1][1]["files_done"], records[1][1]["files_failed"], records[1][1]["documents"]) == ("1", "0", "1")
    # Content-free: no file name and no document text anywhere in the lines.
    joined = " ".join(record.getMessage() for record, _ in records)
    assert "batch.pdf" not in joined and "INVOICE" not in joined and "A-1" not in joined


def test_a_signed_in_run_logs_only_the_recomputed_pseudonym(tmp_path, monkeypatch, caplog) -> None:
    import streamlit as st

    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()

    assert not app.exception
    records = _audit(caplog)
    assert [fields["event"] for _, fields in records] == ["run_started", "run_ended"]
    # Recomputed straight from the documented recipe, not via audit.pseudonym.
    expected = hmac.new(SECRET.encode(), "boss@example.com".strip().lower().encode(), hashlib.sha256).hexdigest()[:12]
    assert all(fields["user"] == expected for _, fields in records)
    joined = " ".join(record.getMessage() for record, _ in records)
    assert "boss@example.com" not in joined and "@" not in joined and "batch.pdf" not in joined


def test_an_allowlist_refusal_is_logged_as_a_warning(tmp_path, monkeypatch, caplog) -> None:
    import streamlit as st

    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="stranger@other.example", email_verified=True))

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()

    assert not app.exception
    assert any("is not on the list" in error.value for error in app.error)
    records = _audit(caplog)
    assert [fields for _, fields in records] == [
        {
            "event": "access_refused",
            "user": audit.pseudonym("stranger@other.example", SECRET),
            "reason": "allowlist",
        }
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]


def test_a_missing_auth_setup_is_logged_as_no_auth(tmp_path, monkeypatch, caplog) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()

    assert not app.exception
    assert any("Sign-in isn't set up" in error.value for error in app.error)
    records = _audit(caplog)
    assert [fields for _, fields in records] == [
        {"event": "access_refused", "user": "unknown", "reason": "no_auth"}
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]


def test_a_top_level_anonymous_secret_is_logged_as_anonymous_secret(tmp_path, monkeypatch, caplog) -> None:
    # _app sets the real variable too: that is what Streamlit does with a
    # top-level string secret, so the app cannot trust the value it sees.
    app = _app(tmp_path, monkeypatch)
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = "1"

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()

    assert not app.exception
    assert any("DOC_PIPELINE_ALLOW_ANONYMOUS" in error.value for error in app.error)
    records = _audit(caplog)
    assert [fields for _, fields in records] == [
        {"event": "access_refused", "user": "unknown", "reason": "anonymous_secret"}
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]


def test_a_server_busy_refusal_is_logged(tmp_path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch)
    slots = RunSlots(running=1, waiting=0)
    assert slots.admit("someone-else") is None  # the server's one place is taken
    monkeypatch.setattr(ui, "_slots", slots)
    data = _pdf(["INVOICE\nInvoice No: A-1"])

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()

    assert not app.exception
    assert any("server is busy" in warning.value for warning in app.warning)
    records = _audit(caplog)
    assert [fields for _, fields in records] == [
        {"event": "run_refused", "user": "anonymous", "reason": "server_busy"}
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]


def test_a_refusal_is_logged_once_per_session_while_the_page_shows_it_every_time(
    tmp_path, monkeypatch, caplog
) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.run()

    assert not app.exception
    assert any("Sign-in isn't set up" in error.value for error in app.error)  # shown again on the rerun
    assert [fields for _, fields in _audit(caplog)] == [
        {"event": "access_refused", "user": "unknown", "reason": "no_auth"}
    ]


def test_a_new_session_logs_its_refusal_again(tmp_path, monkeypatch, caplog) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        _app(tmp_path, monkeypatch, anonymous=False).run()
        _app(tmp_path, monkeypatch, anonymous=False).run()

    assert len(_audit(caplog)) == 2  # deduped per session, so this is no rate limit


def test_two_server_busy_refusals_in_one_session_log_one_line(tmp_path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch)
    slots = RunSlots(running=1, waiting=0)
    assert slots.admit("someone-else") is None
    monkeypatch.setattr(ui, "_slots", slots)
    data = _pdf(["INVOICE\nInvoice No: A-1"])

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()
        app.button(key="start").click().run()

    assert not app.exception
    assert any("server is busy" in warning.value for warning in app.warning)  # still shown the second time
    assert [fields for _, fields in _audit(caplog)] == [
        {"event": "run_refused", "user": "anonymous", "reason": "server_busy"}
    ]


def test_a_lone_surrogate_makes_the_pseudonym_unknown_instead_of_raising(monkeypatch) -> None:
    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "_cookie_secret", lambda: SECRET)
    assert ui._pseudonym_or_unknown("boss@example.com") == audit.pseudonym("boss@example.com", SECRET)
    assert ui._pseudonym_or_unknown("\ud800@example.com") == "unknown"  # in the email claim
    monkeypatch.setattr(ui, "_cookie_secret", lambda: "\ud800")
    assert ui._pseudonym_or_unknown("boss@example.com") == "unknown"  # in the cookie secret


def test_a_pseudonym_that_raises_cannot_break_the_refusal_page(tmp_path, monkeypatch, caplog) -> None:
    import streamlit as st

    def broken(email: str, secret: str | None) -> str:
        raise RuntimeError("no pseudonym today")

    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="stranger@other.example", email_verified=True))
    monkeypatch.setattr(audit, "pseudonym", broken)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()

    assert not app.exception
    assert any("is not on the list" in error.value for error in app.error)
    assert [fields["user"] for _, fields in _audit(caplog)] == ["unknown"]


def test_an_email_claim_with_a_lone_surrogate_still_gets_the_refusal_page(tmp_path, monkeypatch, caplog) -> None:
    import streamlit as st

    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email=chr(0xD800) + "@x.example", email_verified=True))

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()

    assert not app.exception
    assert any("is not on the list" in error.value for error in app.error)
    assert [button.label for button in app.button if button.label == "Sign out"] == ["Sign out"]
    assert [fields for _, fields in _audit(caplog)] == [
        {"event": "access_refused", "user": "unknown", "reason": "allowlist"}
    ]


def test_too_many_files_logs_once_and_not_again_on_a_rerun(tmp_path, monkeypatch, caplog) -> None:
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value([(f"file-{number}.pdf", data, "application/pdf") for number in range(6)]).run()
        records = _audit(caplog)
        app.run()  # a plain rerun of the same page

    assert not app.exception
    assert [fields for _, fields in records] == [
        {"event": "run_refused", "user": "anonymous", "reason": "too_many_files"}
    ]
    assert [record.levelno for record, _ in records] == [logging.WARNING]
    # The rerun added nothing, and no file name reached the line.
    assert _audit(caplog) == records
    assert not any("file-" in record.getMessage() for record, _ in records)


def test_a_sign_out_with_a_job_logs_what_it_stopped(tmp_path, monkeypatch, caplog) -> None:
    import streamlit as st

    from doc_pipeline.app.ui import Job

    monkeypatch.setattr(st, "logout", lambda: None)
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))
    app.session_state["job"] = Job(Future(), _PageCrashesMidRun(), rules_only=True, user="0123456789ab")

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()  # breaks mid-run, leaving the job and its Sign out button
        app.button(key="sign_out").click().run()

    assert not app.exception
    assert "job" not in app.session_state
    assert [fields for _, fields in _audit(caplog)] == [
        {"event": "signout_stopped_run", "user": "0123456789ab"}
    ]


def test_a_failing_audit_call_never_breaks_a_run(tmp_path, monkeypatch) -> None:
    from doc_pipeline.app import audit

    def raising(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no logging today")

    monkeypatch.setattr(audit, "log_event", raising)
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)

    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert "result" in app.session_state
    assert any("Results" in header.value for header in app.subheader)
