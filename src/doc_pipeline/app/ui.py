"""Streamlit page: upload PDFs, choose fields and optional models, press Start."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st

from ..errors import ConfigurationError, RunStopped
from ..pricing import ModelUsage, Price, add_usage, estimate, format_usd
from ..providers import import_provider_sdks
from . import audit
from .access import client_ip, is_allowed, sign_in_providers, user_key
from .hardening import set_non_dumpable
from .session import (
    ModelTestQuota,
    Refusal,
    RunControl,
    RunProgress,
    RunQuota,
    RunResult,
    RunSlots,
    Upload,
    check_models,
    check_uploads,
    cleanup_workspaces,
    ensure_workspace,
    remove_workspace,
    reserve_workspace,
    run_uploads,
    sweep_workspaces,
    touch_workspace,
    upload_refusal_codes,
)
from .settings import (
    DATE_ORDER_LABELS,
    EFFORT_LABELS,
    MAX_SLOTS,
    MAX_WORKERS,
    PROVIDER_LABELS,
    SLOT_TITLES,
    TYPE_LABELS,
    AppChoices,
    AppDefaults,
    ModelSlot,
    build_app_config,
    describe_mode,
    effort_options,
    load_app_defaults,
    preset_rows,
)

logger = logging.getLogger(__name__)

RULES_ONLY_WARNING = (
    "**No API key entered.** Documents are split with the boundary rules and fields are filled by matching "
    "labels in the PDF text. Results may be less accurate than with an AI model, so check them before use."
)
OTHER_MODEL = "Other…"
EFFORT_HELP = {
    "gemini": "Thinking level. Off works on Gemini 2.5 Flash; Gemini 3 models use Minimal as their lowest. "
    "Higher effort is slower and costs more.",
    "openai": "reasoning_effort for reasoning models (o-series, GPT-5 family). Keep Provider default for models "
    "such as gpt-4o-mini, which do not accept it.",
    "glm": "Turns GLM thinking mode on or off (GLM-4.5 and later).",
}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# How often the progress bar is redrawn while a run works in the background.
POLL_SECONDS = 0.5
APP_TITLE = "PDF Splitter & Extractor"
# Set to 1 to use the app without sign-in: local development and tests only.
ANONYMOUS_ENV = "DOC_PIPELINE_ALLOW_ANONYMOUS"
PROVIDER_NAMES = {"google": "Google", "microsoft": "Microsoft"}
# Two people behind one office or carrier NAT share one IP-keyed run slot; the
# refusal names that and points at the account-keyed way out.
NAT_RUN_REFUSAL = (
    "Your network already has a run in progress (limits count per IP address). Sign in, or wait for it to finish."
)
# The public anonymous tier counts per IP address, so a visit that arrives
# without a usable one has nothing stable to count and is refused instead of
# keyed by a session a reload can change. Where sign-in is set up the refusal
# points at it; where it is not, there is no way in to name.
NO_CLIENT_IP_REFUSAL = Refusal(
    "Run limits count per IP address, but your visit arrived without a usable one. Please sign in to use the app.",
    "no_client_ip",
)
NO_CLIENT_IP_WITHOUT_SIGN_IN_REFUSAL = Refusal(
    "Run limits count per IP address, but your visit arrived without a usable one. "
    "Runs cannot be started from this visit.",
    "no_client_ip",
)
# One line for the owner: an [access] allowlist checks signed-in accounts
# only, so it does not narrow the open anonymous tier.
ALLOWLIST_ANONYMOUS_WARNING = (
    "[access] allow_anonymous is on together with an allowlist (allowed_emails or allowed_domains): the allowlist "
    "does not restrict anonymous visitors, so the app is open to anyone who is not signed in."
)


@dataclass
class Job:
    """A run working in a background thread, kept across page reruns."""

    future: Future
    progress: RunProgress
    rules_only: bool
    control: RunControl = field(default_factory=RunControl)
    # The audit pseudonym of the account that started the run, carried to the
    # run_ended and signout_stopped_run lines.
    user: str = "unknown"
    started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class Visitor:
    """Who is using the page: a signed-in account, or anyone when sign-in is off."""

    # Issuer and subject of the account; None when anonymous.
    key: str | None = None
    email: str = ""
    # Provider names st.login() can use, when sign-in is set up; None when it
    # is not. Set for an anonymous visitor too, so the page can offer sign-in.
    providers: list[str] | None = None


def main(config_path: Path) -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    _harden_process()
    _import_provider_sdks()
    defaults = _load_defaults(str(config_path))
    visitor = _visitor(defaults.source_url)
    if visitor is None:
        return
    audit_user = _audit_user(visitor)
    if visitor.key is None:
        # A note for the owner, once per process: an open anonymous tier is
        # not narrowed by an [access] allowlist.
        _warn_allowlist_gap()
    cleanup_workspaces(defaults.limits.workspace_ttl_minutes)
    _start_cleanup(defaults.limits.workspace_ttl_minutes)
    workspace = _workspace()
    # Run limits count signed-in accounts and, when anonymous, the client IP
    # (the validated last X-Forwarded-For entry; the session fallback is the
    # local environment switch's alone, and a public visit without a usable
    # IP is refused instead).
    ip = _client_ip(st.context.headers)
    user = _limit_key(visitor, workspace, ip)
    slots = _run_slots(defaults.limits.max_concurrent_runs, defaults.limits.max_waiting_runs)
    quota = _run_quota(defaults.limits.max_runs_per_hour, defaults.limits.max_runs_per_day)
    test_quota = _model_test_quota(defaults.limits.max_model_tests_per_hour)

    st.title(APP_TITLE)
    st.caption(
        "Upload PDF batches, choose the fields you want, and press Start. Each batch is split into separate "
        "documents and the fields are written to one Excel workbook."
    )
    _notice(defaults, visitor, user)
    _account(visitor, workspace, user)
    choices, test_clicked, model_area = _sidebar(defaults)
    _source_offer(defaults.source_url)
    if choices.active_slots():
        st.info(describe_mode(choices), icon="🤖")
    else:
        st.warning(RULES_ONLY_WARNING, icon="⚠️")

    st.subheader("1. Fields to extract")
    choices.fields = _fields_editor(defaults)

    st.subheader("2. Upload PDFs")
    limits = defaults.limits
    files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Up to {limits.max_files} files per run, {limits.max_file_mb} MB and {limits.max_pages} pages each.",
    )
    uploads = [Upload(file.name, file.getvalue()) for file in files or []]
    upload_problems = check_uploads(uploads, limits)
    # The audit line carries codes, the page the messages that name files.
    refusal_codes = upload_refusal_codes(uploads, limits)
    if refusal_codes:
        if refusal_codes != st.session_state.get("upload_refusal"):
            _log("run_refused", user=audit_user, warning=True, reason=",".join(refusal_codes))
            st.session_state["upload_refusal"] = refusal_codes
    else:
        st.session_state.pop("upload_refusal", None)
    if uploads and not upload_problems:
        st.caption(f"{len(uploads)} PDF file(s) ready. Nothing is processed until you press Start.")

    _options(choices)

    try:
        config, api_keys = build_app_config(defaults, choices, workspace)
        config_error = None
    except ConfigurationError as exc:
        config, api_keys, config_error = None, {}, str(exc)
    if config_error:
        st.error(config_error)
    for problem in upload_problems:
        st.error(problem)

    if test_clicked:
        with model_area:
            if user is None:
                _refuse_no_client_ip(audit_user)
            elif config is None:
                # The settings are checked before the quota is charged: a
                # press that cannot run a test uses up none of the
                # model-test limits (_test_models shows the same refusal).
                _test_models(config, api_keys, config_error)
            else:
                # The test quota is charged alone: a model test never uses up
                # the run limits.
                refusal = test_quota.admit(user)
                if refusal is None:
                    _test_models(config, api_keys, config_error)
                else:
                    _log_once("run_refused", refusal.code, user=audit_user)
                    st.warning(refusal)

    st.subheader("3. Process")
    job: Job | None = st.session_state.get("job")
    start = st.button(
        "Start Processing",
        type="primary",
        key="start",
        disabled=job is not None or config is None or not uploads or bool(upload_problems),
    )
    if start and config is not None and job is None:
        # The no-key refusal comes first, so a None key reaches neither the
        # slots, the run quota, nor the test quota.
        refusal = _no_client_ip_refusal() if user is None else _refuse_run(slots, quota, user)
        if refusal:
            _log_once("run_refused", refusal.code, user=audit_user)
            st.warning(refusal)
        else:
            st.session_state.pop("result", None)
            job = st.session_state["job"] = _start_job(
                config, api_keys, uploads, workspace, slots, user, defaults.limits.run_timeout_minutes, audit_user
            )
            started = {
                "files": len(uploads),
                "bytes": sum(len(upload.data) for upload in uploads),
                "mode": "models" if config.models else "rules",
            }
            if config.models:
                # Which AI providers a run may reach, in order, without repeats.
                started["providers"] = ",".join(dict.fromkeys(model.provider for model in config.models))
            _log("run_started", user=audit_user, **started)
    if job is not None:
        _wait_for(job)
    error = st.session_state.pop("run_error", None)
    if error:
        st.error(error)

    result = st.session_state.get("result")
    if result is not None:
        _results(result, bool(st.session_state.get("rules_only")), workspace, defaults.prices)


def _refuse_run(slots: RunSlots, quota: RunQuota, user: str) -> Refusal | None:
    """Why the run may not start, or None when a place was held for it.

    The one-run rule holds the place first; when the quota then refuses, the
    place is given back, so a run that never starts is charged to neither
    limit. On an `ip:` key, "you already have a run" is usually someone else
    on the same network, so it is reworded and points at sign-in.
    """
    refusal = slots.admit(user)
    if refusal is None:
        refusal = quota.admit(user)
        if refusal is not None:
            slots.leave(user)
    if refusal is not None and refusal.code == "user_active" and user.startswith("ip:"):
        refusal = Refusal(NAT_RUN_REFUSAL, refusal.code)
    return refusal


def _client_ip(headers: Any) -> str | None:
    """The client IP from every X-Forwarded-For line, or None for none.

    A proxy that appends the client address as a second X-Forwarded-For line
    makes a plain mapping lookup return only the first line, whose earlier
    entries the client chose; get_all sees every line, so the last entry is
    the proxy's own. StreamlitHeaders.get_all returns [] for a missing header,
    so the join stays safe; a plain dict (as in the tests) has no get_all and
    is used as it is.
    """
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        return client_ip({"X-Forwarded-For": ",".join(get_all("X-Forwarded-For"))})
    return client_ip(headers)


def _limit_key(visitor: Visitor, workspace: Path, ip: str | None) -> str | None:
    """The key the run and model-test limits count this visit under, or None to refuse.

    A signed-in account; else the client IP when one arrived; else the
    session's workspace - but that last fallback only under the local
    environment switch, whose visitors are trusted. On the public
    `[access] allow_anonymous` tier a visit without a usable IP has nothing
    stable to count against (a reload would make a new session, with fresh
    limits), so None means the caller refuses it.
    """
    if visitor.key is not None:
        return visitor.key
    if ip:
        return f"ip:{ip}"
    if _anonymous_allowed():
        return f"session:{workspace}"
    return None


def _no_client_ip_refusal() -> Refusal:
    """The no-usable-IP refusal, pointing at sign-in only where it is set up."""
    return NO_CLIENT_IP_REFUSAL if _sign_in_set_up() else NO_CLIENT_IP_WITHOUT_SIGN_IN_REFUSAL


def _refuse_no_client_ip(audit_user: str) -> None:
    """Show the no-usable-IP refusal and log it once per session."""
    _log_once("run_refused", "no_client_ip", user=audit_user)
    st.warning(_no_client_ip_refusal())


def _start_job(
    config,
    api_keys: dict[str, str],
    uploads: list[Upload],
    workspace: Path,
    slots: RunSlots,
    user: str,
    timeout_minutes: int | None = None,
    audit_user: str = "unknown",
) -> Job:
    """Run in a background thread, so the page can show live progress.

    The run waits for one of the server's run slots; slots.admit(user) must
    have succeeded. While queued it keeps the workspace fresh and can give up
    when cancelled. A click during the run reruns the page, and the rerun
    picks the job up again instead of stopping it.
    """
    progress = RunProgress()
    control = RunControl(timeout_minutes)
    work = partial(run_uploads, config, api_keys, uploads, workspace, progress, control)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="doc-pipeline-run")
    try:
        # The folder is created here and only here, inside the guard that
        # gives the slot back if it fails: a page load that never starts a
        # run leaves none behind (see _workspace).
        ensure_workspace(workspace)
        future = pool.submit(
            slots.run,
            user,
            work,
            progress.wait,
            keep_alive=partial(touch_workspace, workspace),
            abort=control.cancelled.is_set,
        )
    except BaseException:
        slots.leave(user)
        raise
    pool.shutdown(wait=False)  # the thread ends with the run
    return Job(future, progress, rules_only=not config.models, control=control, user=audit_user)


def _wait_for(job: Job) -> None:
    st.button("Cancel run", key="cancel_run", on_click=job.control.cancelled.set)
    bar = st.progress(0.0, text="Starting…")
    note = st.empty()
    while not job.future.done():
        fraction, text, retry = job.progress.snapshot()
        bar.progress(fraction, text=text)
        if retry:
            note.caption(retry)
        else:
            note.empty()
        time.sleep(POLL_SECONDS)
    st.session_state.pop("job", None)
    result: RunResult | None = None
    failure: BaseException | None = None
    try:
        result = job.future.result()
    except RunStopped as exc:
        failure = exc
        st.session_state["run_error"] = f"The run ended early: {exc}."
    except Exception as exc:
        failure = exc
        st.session_state["run_error"] = f"Processing stopped unexpectedly ({type(exc).__name__}). Please try again."
    else:
        if result is None:  # cancelled while queued: the work never started
            st.session_state["run_error"] = "The run was cancelled while waiting for a free slot."
        else:
            st.session_state["result"] = result
            st.session_state["rules_only"] = job.rules_only
            # Added once per run (the job is removed above); Start over keeps it,
            # since that money is spent.
            st.session_state["session_usage"] = add_usage(st.session_state.get("session_usage", {}), result.model_usage)
    ended: dict[str, object] = {
        "outcome": _outcome(result, failure, job.control),
        "seconds": int(time.monotonic() - job.started),
    }
    if result is not None:
        summary = result.summary
        # files_done: the files that produced something; the failed ones are
        # counted apart, since a stopped run reports the rest as not run.
        ended.update(
            files_done=summary["files"] - summary["failed"],
            files_failed=summary["failed"],
            documents=summary["documents"],
        )
    _log("run_ended", user=job.user, **ended)
    # Rerun so the Start button and the results reflect the finished run.
    st.rerun()


def _log(event: str, *, user: str = "unknown", warning: bool = False, **fields: object) -> None:
    """One audit line, best-effort: logging must never break a run or the page.

    audit.log_event already swallows its own errors; this wrapper catches the
    case where the module itself is broken or patched out.
    """
    try:
        audit.log_event(event, user=user, warning=warning, **fields)
    except Exception:
        pass


def _log_once(event: str, reason: str, *, user: str = "unknown") -> None:
    """A refusal line, at most once per session per event and reason code.

    The page shows the refusal every time; only the log line is deduped, so a
    visitor who keeps pressing a button does not flood the log. A new session
    (a reload, a new tab) logs again, so this is no rate limit.
    """
    try:
        logged: set[tuple[str, str]] = st.session_state.setdefault("audit_logged", set())
        if (event, reason) in logged:
            return
        logged.add((event, reason))
    except Exception:
        pass  # no session state: log the line rather than lose it
    _log(event, user=user, warning=True, reason=reason)


def _pseudonym_or_unknown(email: str) -> str:
    """The audit pseudonym for an email, or `unknown` when anything about it is off.

    A lone surrogate in the email claim or the cookie secret makes the HMAC
    raise; the refusal page must still render.
    """
    try:
        return audit.pseudonym(email, _cookie_secret())
    except Exception:
        return "unknown"


def _outcome(result: RunResult | None, failure: BaseException | None, control: RunControl) -> str:
    """Why a run ended, as the one outcome code the audit log knows.

    A cancel while queued never started, so it is plain `cancelled`. A stop
    after the run began - a RunStopped, or a result reporting one - is named
    by the job's own control flags (see _stop). Anything else that raised is
    `failed`, and a result that stopped nowhere is `finished`.
    """
    if failure is not None:
        return _stop(control) if isinstance(failure, RunStopped) else "failed"
    if result is None:
        return "cancelled"
    if result.stopped is None:
        return "finished"
    return _stop(control)


def _stop(control: RunControl) -> str:
    """Which stop a begun run hit, from the flags that stop set."""
    if control.workspace_removed.is_set():
        return "signout"
    if control.cancelled.is_set():
        return "cancelled"
    return "timed_out"


# Server-wide state, held in module globals rather than st.cache_resource:
# any websocket client, signed in or not, can send a clear_cache message, and
# a cleared cache would forget who has a run (a fresh RunSlots sidesteps
# max_concurrent_runs and the one-run-per-user rule, a fresh RunQuota the
# hourly and daily caps) and start a second cleanup thread on the next page
# load. A module global survives that clear; the lock keeps the
# once-per-process creation safe when page reruns overlap in threads.
# _hardened, _sdks_imported and _anonymous are None until first set and then
# kept (a False is not retried); _anonymous_refused is re-read while it is
# not True, so it latches once a secret is seen.
_INIT_LOCK = threading.Lock()
_defaults: dict[str, AppDefaults] = {}
_slots: RunSlots | None = None
_quota: RunQuota | None = None
_test_quota: ModelTestQuota | None = None
_hardened: bool | None = None
_sdks_imported: bool | None = None
_sweeper: threading.Thread | None = None
_anonymous: bool | None = None
_anonymous_refused: bool | None = None
_allowlist_warned: bool | None = None


def _load_defaults(config_path: str) -> AppDefaults:
    global _defaults
    with _INIT_LOCK:
        defaults = _defaults.get(config_path)
        if defaults is None:
            defaults = _defaults[config_path] = load_app_defaults(Path(config_path))
        return defaults


def _run_slots(running: int, waiting: int) -> RunSlots:
    """One set of run slots, shared by every session on this server.

    The arguments of the first call win; later calls return the existing object.
    """
    global _slots
    with _INIT_LOCK:
        if _slots is None:
            _slots = RunSlots(running, waiting)
        return _slots


def _run_quota(per_hour: int, per_day: int) -> RunQuota:
    """One run quota, shared by every session on this server.

    The arguments of the first call win; later calls return the existing object.
    """
    global _quota
    with _INIT_LOCK:
        if _quota is None:
            _quota = RunQuota(per_hour, per_day)
        return _quota


def _model_test_quota(per_hour: int) -> ModelTestQuota:
    """One model-test quota, shared by every session on this server.

    The arguments of the first call win; later calls return the existing object.
    """
    global _test_quota
    with _INIT_LOCK:
        if _test_quota is None:
            _test_quota = ModelTestQuota(per_hour)
        return _test_quota


def _warn_allowlist_gap() -> None:
    """One WARNING per process when an allowlist coexists with the open tier.

    The allowlist checks signed-in accounts only - an anonymous visitor has no
    email to check it against - so with `[access] allow_anonymous` on it does
    not narrow who may use the app. The switch and the allowlist are re-read
    on every page load, but the warning itself is for the owner reading the
    logs, so it latches instead of repeating once per visitor.
    """
    global _allowlist_warned
    if not _anonymous_access():
        return
    access = _secrets("access")
    if not (access.get("allowed_emails") or access.get("allowed_domains")):
        return
    with _INIT_LOCK:
        if _allowlist_warned:
            return
        _allowlist_warned = True
    logger.warning(ALLOWLIST_ANONYMOUS_WARNING)


def _harden_process() -> bool:
    """Make this server process non-dumpable, once per process.

    A same-uid process - a render child gone wrong, say - then can neither
    ptrace the server nor read its ptrace-gated /proc files, where the
    sessions' API keys sit in memory (see hardening). main() reruns on every
    page load, so the call must run only the first time (see _INIT_LOCK).
    """
    global _hardened
    with _INIT_LOCK:
        if _hardened is None:
            _hardened = set_non_dumpable()
            if not _hardened and sys.platform.startswith("linux"):
                logger.warning("Server hardening failed: the process is still dumpable.")
        return _hardened


def _import_provider_sdks() -> bool:
    """Import the provider SDKs once per process, at the first page load.

    A compromised render child runs as this process's user, so it could edit
    the SDK code in the venv. Importing the SDKs here pins their top-level
    modules (those that imported successfully) to what was on disk at
    start-up - a page rerun does not re-import modules already in
    sys.modules - but that only narrows what a changed SDK can reach, it is
    not a boundary: the child can force a live reload through the
    always-polled .streamlit/config.toml and .streamlit/secrets.toml, so its
    writes to server code or config can take effect without a restart, and
    anything imported later - the OpenAI SDK loads its request code
    (openai.resources.*) at the first OpenAI or GLM request, not here - could
    still load modified code. The real fix, a separate user or a Landlock
    policy for the render child, is not done. main() reruns on every page
    load, so the import must run only the first time (see _INIT_LOCK).
    """
    global _sdks_imported
    with _INIT_LOCK:
        if _sdks_imported is None:
            import_provider_sdks()
            _sdks_imported = True
        return _sdks_imported


def _anonymous_allowed() -> bool:
    """Whether anonymous use is switched on, read once per process.

    DOC_PIPELINE_ALLOW_ANONYMOUS is meant to be a real environment variable,
    but Streamlit copies top-level string secrets into os.environ whenever it
    re-reads secrets.toml, so the value this process sees can be planted or
    changed while the server runs. Reading the variable at the first page
    load and keeping the answer means a later secrets edit cannot turn
    sign-in off mid-flight; anonymous use (only) is refused once a top-level
    secret of that name is seen (see _anonymous_secret_seen). main() reruns on
    every page load, so the read must run only the first time (see
    _INIT_LOCK).
    """
    global _anonymous
    with _INIT_LOCK:
        if _anonymous is None:
            _anonymous = os.environ.get(ANONYMOUS_ENV) == "1"
        return _anonymous


def _anonymous_secret_seen() -> bool:
    """Whether a top-level ANONYMOUS_ENV key was ever seen in the secrets.

    Such a key means someone tried to set the switch in secrets.toml, where
    Streamlit promotes top-level string values into os.environ: the
    environment variable this process then carries cannot be told apart from
    a real setting, so anonymous use stays refused for the rest of the
    process, even if the key is later removed or auth is set up - and only
    anonymous use, in every branch: sign-in is not refused (see _visitor).
    Any value type counts, since a value that is not a string today can be
    edited into one (and the string would be copied into the environment).
    """
    global _anonymous_refused
    with _INIT_LOCK:
        if not _anonymous_refused:
            _anonymous_refused = _secret_present(ANONYMOUS_ENV)
        return bool(_anonymous_refused)


def _anonymous_access() -> bool:
    """Whether the secrets switch `[access] allow_anonymous` is on.

    Only a literal TOML `true` counts: a missing section, `false`, or a string
    like "true" all mean off, so the tier fails closed. Read on every page
    load like the allowlist (unlike the once-per-process environment switch),
    since a plain mapping value carries none of its promotion risks.
    """
    access = _secrets("access")
    return access is not None and access.get("allow_anonymous") is True


def _sign_in_set_up() -> bool:
    """Whether an [auth] section exists, so sign-in can be offered to visitors."""
    # `is not None`, not truthiness: [] means one provider set directly under
    # [auth], which is still sign-in set up.
    return sign_in_providers(_secrets("auth")) is not None


def _start_cleanup(ttl_minutes: int) -> threading.Thread:
    """One background thread per server process sweeping expired workspaces.

    Runs between page loads too, so folders are deleted even when nobody
    visits. main() reruns on every page load, so it must not start the
    thread itself (see _INIT_LOCK). The arguments of the first call win; later
    calls return the existing thread.
    """
    global _sweeper
    with _INIT_LOCK:
        if _sweeper is None:
            stop = threading.Event()
            sweeper = threading.Thread(
                target=sweep_workspaces, args=(ttl_minutes, stop), name="doc-pipeline-cleanup", daemon=True
            )
            sweeper.start()
            _sweeper = sweeper
        return _sweeper


def _visitor(source_url: str) -> Visitor | None:
    """Ask for sign-in until the visitor may use the app; None stops the page.

    Without an [auth] secrets section the app refuses to run, unless
    DOC_PIPELINE_ALLOW_ANONYMOUS=1 (local development and tests) or the
    secrets switch `[access] allow_anonymous` is on. That environment switch
    is read once per process, before the secrets are looked at, and a
    top-level secret of that name refuses anonymous use for the rest of the
    process (see _anonymous_allowed and _anonymous_secret_seen). With sign-in
    set up, a signed-out visitor is admitted anonymously only by the secrets
    switch, and never once that latch is set, and keeps the sign-in offer in
    the sidebar (see _account); the allowlist stays a check on signed-in
    accounts. Every page state that stops here also shows the AGPL source
    offer (see _source_offer), so the licence is named even before sign-in
    works.
    """
    # Both reads run before the sign-in, on every page load: the first one
    # captures the variable before any secrets are looked at, and the second
    # must see a top-level secret whenever it appears, so the refusal below
    # stays latched for the process (see the two functions).
    anonymous = _anonymous_allowed()
    refused = _anonymous_secret_seen()
    providers = sign_in_providers(_secrets("auth"))
    if providers is None:
        if refused:
            _log_once("access_refused", "anonymous_secret")
            st.title(APP_TITLE)
            st.error(
                f"{ANONYMOUS_ENV} is set in this app's secrets, not in its environment. Streamlit copies top-level "
                "secrets into the environment, so the app cannot treat its value as a real environment setting, and "
                "anonymous use stays off. Set the variable in the environment for local development, or set up an "
                "[auth] section in the secrets (see the README)."
            )
            _source_offer(source_url)
            return None
        if anonymous or _anonymous_access():
            return Visitor()
        _log_once("access_refused", "no_auth")
        st.title(APP_TITLE)
        st.error(
            "Sign-in isn't set up for this app, so it can't be used yet. The app's owner needs to add an "
            "[auth] section to its secrets, or open it to visitors without sign-in with `allow_anonymous = true` "
            "under `[access]` (see the README)."
        )
        _source_offer(source_url)
        return None
    # .get: a missing is_logged_in (no auth context yet) counts as signed out.
    if not st.user.get("is_logged_in"):
        # The latch refuses anonymous use in every branch: as in the no-auth
        # case above, a top-level ANONYMOUS_ENV secret keeps the secrets
        # switch from opening the tier (see _anonymous_secret_seen).
        if _anonymous_access() and not refused:
            # No identity to check the allowlist against, so the allowlist
            # does not apply here; it keeps guarding signed-in accounts.
            return Visitor(providers=providers)
        st.title(APP_TITLE)
        st.info("Sign in to upload PDFs and extract their fields.")
        for provider in providers or [None]:
            name = PROVIDER_NAMES.get(provider, provider.title()) if provider else ""
            st.button(
                f"Sign in with {name}" if name else "Sign in",
                key=f"sign_in_{provider or 'default'}",
                on_click=st.login,
                args=(provider,) if provider else (),
            )
        _source_offer(source_url)
        return None
    claims = st.user.to_dict()
    email = str(claims.get("email") or "")
    if not is_allowed(claims, _secrets("access")):
        _log_once("access_refused", "allowlist", user=_pseudonym_or_unknown(email))
        st.title(APP_TITLE)
        # A lone surrogate in the claim cannot be encoded for the browser, so it is replaced.
        shown = email.encode("utf-8", "replace").decode("utf-8")
        st.error(f"{shown or 'This account'} is not on the list of people who can use this app.")
        st.button("Sign out", on_click=st.logout)
        _source_offer(source_url)
        return None
    return Visitor(key=user_key(claims), email=email)


def _cookie_secret() -> str | None:
    """The [auth] cookie_secret, when it is a string: audit pseudonyms key on it."""
    auth = _secrets("auth")
    if not auth:
        return None
    secret = auth.get("cookie_secret")
    return secret if isinstance(secret, str) else None


def _audit_user(visitor: Visitor) -> str:
    """The pseudonymous id the audit lines carry for this visitor.

    Never the email itself: a hash keyed with the cookie secret, `anonymous`
    without sign-in, `unknown` when anything about the secrets is off.
    """
    try:
        if visitor.key is None:
            return "anonymous"
        return audit.pseudonym(visitor.email, _cookie_secret())
    except Exception:
        return "unknown"


def _secrets(section: str) -> Mapping[str, Any] | None:
    """One section of the app's secrets, or None when it (or the secrets file) is missing."""
    try:
        value = st.secrets.get(section)
    except Exception:  # no secrets file on this host
        return None
    return value if isinstance(value, Mapping) else None


def _secret_present(name: str) -> bool:
    """Whether the app's secrets have a top-level entry of this name."""
    try:
        return st.secrets.get(name) is not None
    except Exception:  # no secrets file on this host
        return False


def _anonymous_caption(user: str | None) -> str:
    """The sidebar line naming what an anonymous visit's run limits count per."""
    if user is None:
        # The way out is sign-in where it exists; where it does not, this
        # visit simply cannot start runs.
        if _sign_in_set_up():
            return "Not signed in: no usable IP address arrived, so starting a run needs sign-in."
        return "Not signed in: no usable IP address arrived, so runs cannot be started from this visit."
    if user.startswith("session:"):
        return "Not signed in: run limits count per browser session."
    return "Not signed in: run limits count per IP address, shared with others on your network."


def _account(visitor: Visitor, workspace: Path, user: str | None) -> None:
    if visitor.key is None:
        with st.sidebar:
            # What the limits count per, whatever opened the tier: the caption
            # carries the key kind even where no sign-in is set up to offer.
            st.caption(_anonymous_caption(user))
            # `is not None`, not truthiness: [] means one provider set directly
            # under [auth], which is still sign-in set up and still needs its offer.
            offers = (visitor.providers or [None]) if visitor.providers is not None else []
            for provider in offers:
                name = PROVIDER_NAMES.get(provider, provider.title()) if provider else ""
                st.button(
                    f"Sign in with {name}" if name else "Sign in",
                    key=f"sign_in_{provider or 'default'}",
                    on_click=st.login,
                    args=(provider,) if provider else (),
                )
        return
    with st.sidebar:
        st.caption(f"Signed in as {visitor.email or 'your account'}")
        st.button(
            "Sign out",
            key="sign_out",
            on_click=_sign_out,
            args=(workspace,),
            help="Also forgets your API keys and deletes your files from the server.",
        )


def _sign_out(workspace: Path) -> None:
    job = st.session_state.get("job")
    if job is not None:
        _log("signout_stopped_run", user=job.user)
        # Stop queued or running work before its folder goes, and mark the
        # folder as removed so the run leaves nothing behind at its path.
        job.control.cancelled.set()
        job.control.workspace_removed.set()
    remove_workspace(workspace)
    st.session_state.clear()
    st.logout()


def _workspace() -> Path:
    """This session's folder path, reserved once and created only for a run.

    A bare page load creates nothing: the folder is made (ensure_workspace)
    once Start's refusals have passed, so a scripted visit that never uploads
    or runs leaves no folder for cleanup to sweep.
    """
    path = st.session_state.get("workspace")
    if not path:
        path = st.session_state["workspace"] = str(reserve_workspace())
    # Refresh the folder's timestamp so cleanup does not sweep it mid-run
    # when a full queue keeps the visitor waiting (touch_workspace allows a
    # missing path, so a bare page load still creates nothing).
    touch_workspace(Path(path))
    return Path(path)


def _notice(defaults: AppDefaults, visitor: Visitor, user: str | None) -> None:
    # "No sign-in" describes the host, so it belongs only where the anonymous
    # tier is open (the environment switch without sign-in set up, or the
    # secrets switch) - not on a host where signing in is required.
    open_tier = _anonymous_access() or (not _sign_in_set_up() and _anonymous_allowed())
    lead = "**No sign-in is needed on this host**, but runs" if open_tier else "Runs"
    if visitor.key is None and user is not None and user.startswith("session:"):
        counted = "counted per browser session when you are not signed in, and per account when you are"
    elif visitor.key is None and user is None:
        # Where no sign-in is set up there is no signing in to name.
        if _sign_in_set_up():
            counted = (
                "counted per account when you are signed in (this visit arrived without a usable IP address, "
                "so it needs to sign in to start runs)"
            )
        else:
            counted = (
                "counted per account when you are signed in (this visit arrived without a usable IP address, "
                "so it cannot start runs)"
            )
    else:
        counted = (
            "counted per IP address when you are not signed in (shared with everyone else on that network) "
            "and per account when you are"
        )
    with st.expander("Privacy and accuracy (beta)"):
        st.markdown(
            f"""
- This is a beta. Please don't upload confidential documents.
- **Your API keys** are sent to this server and kept in its memory for your session only. They are never
  written to disk or logs, and they are dropped when your session ends or you sign out. Use a key made for this
  app, with a spending limit, and delete it when you're done.
- **With API keys**, pages are sent to the AI provider you choose (Google, OpenAI, or Z.ai / Zhipu; the
  BigModel endpoint is in China), under your account's terms and billed to you. On Gemini's free tier, Google
  may use what you send to improve its products, and people may review it.
- **Without API keys**, everything runs on this server and nothing is sent to an AI provider, but results are
  less accurate.
- **Check before you pay:** a document from someone else can hide text meant to mislead the AI. Confirm bank
  details with the vendor before paying.
- Uploaded files and results are deleted when you press Start over or sign out, or automatically after
  {defaults.limits.workspace_ttl_minutes} minutes.
- {lead} are limited to {defaults.limits.max_runs_per_hour} per hour
  and {defaults.limits.max_runs_per_day} per day, {counted}. Only **Start Processing** counts against those
  limits; **Test models** has its own hourly one.
- Duplicate checks compare only the files you upload in this session.
"""
        )


def _source_offer(source_url: str) -> None:
    """The AGPL section 13 source offer: a small sidebar line on every page state."""
    with st.sidebar:
        if source_url:
            # Percent-encoded, so no ( ) [ ] < > or backslash in the address can break the link.
            target = quote(source_url, safe=":/@!$&'*+,;=%")
            st.caption(f"[Source code (AGPL-3.0)]({target})")
        else:
            st.caption("Licensed under the GNU AGPL-3.0")


def _sidebar(defaults: AppDefaults) -> tuple[AppChoices, bool, st.delta_generator.DeltaGenerator]:
    slots: list[ModelSlot] = []
    with st.sidebar:
        st.header("AI models (optional)")
        st.caption(
            "Add up to 3 of your own API keys. Model 1 reads every document; Model 2 is asked only when Model 1's "
            "answer can't be confirmed from the PDF text; Model 3 settles disagreements. Leave all empty to run "
            "on rules only."
        )
        for index in range(MAX_SLOTS):
            with st.expander(SLOT_TITLES[index], expanded=index == 0):
                slots.append(_slot(index, defaults))
        active = [number for number, slot in enumerate(slots, start=1) if slot.provider]
        splitting = st.radio(
            "Page splitting",
            ["rules", *active],
            index=1 if active else 0,
            format_func=lambda value: "Rules only" if value == "rules" else f"Rules, then Model {value} for unclear pages",
            help="The rules read page numbers (\"Page 1 of 3\") and title changes; a model is asked only about "
            "pages the rules can't decide.",
            # The key keeps the choice when models are added or removed; it changes
            # only when the first model appears, so Model 1 becomes the default then.
            key="splitting_models" if active else "splitting_none",
        )
        second_opinion = "when_needed"
        if len(active) >= 2:
            second_opinion = st.radio(
                "Second opinion",
                ["when_needed", "always"],
                format_func=lambda value: "Only when needed (cheaper)" if value == "when_needed" else "Always",
                help="Only when needed: Model 2 runs when Model 1's values are not all printed in the PDF text.",
            )
        workers = defaults.extraction_workers
        if active:
            workers = st.slider(
                "Documents read at once",
                1,
                MAX_WORKERS,
                value=defaults.extraction_workers,
                key="workers",
                help="Higher is faster. Lower it if you see rate-limit (429) errors, for example on a free API tier.",
            )
        orders = list(DATE_ORDER_LABELS)
        date_order = st.selectbox(
            "How dates are written",
            orders,
            index=orders.index(defaults.date_order) if defaults.date_order in orders else 0,
            format_func=DATE_ORDER_LABELS.get,
        )
        test_clicked = st.button("Test models", disabled=not active, help="Sends one tiny request per model.")
        model_area = st.container()
    choices = AppChoices(
        slots=slots, splitting=splitting, second_opinion=second_opinion, date_order=date_order, workers=workers
    )
    return choices, test_clicked, model_area


def _slot(index: int, defaults: AppDefaults) -> ModelSlot:
    provider = st.selectbox(
        "Provider",
        ["none", *PROVIDER_LABELS],
        format_func=lambda value: "None" if value == "none" else PROVIDER_LABELS[value],
        key=f"provider_{index}",
    )
    if provider == "none":
        return ModelSlot()
    api_key = st.text_input(
        "API key",
        type="password",
        key=f"api_key_{index}_{provider}",
        help="Used only for this session and never saved.",
    )
    suggestions = [*defaults.model_suggestions.get(provider, ()), OTHER_MODEL]
    picked = st.selectbox("Model", suggestions, key=f"model_{index}_{provider}")
    model = st.text_input("Model ID", key=f"custom_model_{index}_{provider}") if picked == OTHER_MODEL else picked
    effort = st.selectbox(
        "Reasoning effort",
        effort_options(provider),
        format_func=lambda value: EFFORT_LABELS.get(value, value),
        key=f"effort_{index}_{provider}",
        help=EFFORT_HELP[provider],
    )
    base_url = None
    if provider == "glm" and defaults.glm_endpoints:
        endpoint = st.selectbox(
            "Endpoint",
            list(defaults.glm_endpoints),
            key=f"endpoint_{index}",
            help="Z.ai and BigModel keys belong to separate accounts.",
        )
        base_url = defaults.glm_endpoints[endpoint]
    return ModelSlot(provider=provider, api_key=api_key, model=model, effort=effort, base_url=base_url)


def _fields_editor(defaults: AppDefaults) -> list[dict]:
    st.caption(
        "Tick the fields you want as Excel columns, edit them, or add rows for your own. Keywords are the labels "
        "printed next to a value (for example \"Invoice No\"); they matter most when no API key is entered."
    )
    if "field_rows" not in st.session_state:
        st.session_state["field_rows"] = pd.DataFrame(preset_rows(defaults))
    edited = st.data_editor(
        st.session_state["field_rows"],
        key="fields_editor",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", default=True),
            "Field": st.column_config.TextColumn("Field", required=True, max_chars=40),
            "Type": st.column_config.SelectboxColumn("Type", options=list(TYPE_LABELS), default="Text", required=True),
            "Required": st.column_config.CheckboxColumn(
                "Required", default=False, help="Flag the document when this value is missing."
            ),
            "Duplicate key": st.column_config.CheckboxColumn(
                "Duplicate key",
                default=False,
                help="Documents whose duplicate-key values all match are flagged as possible duplicates.",
            ),
            "Keywords": st.column_config.TextColumn(
                "Keywords", help="Comma-separated labels printed next to the value, e.g. Invoice No, Inv #"
            ),
            "Description": st.column_config.TextColumn("Description (for the AI)"),
            "Must appear in text": st.column_config.CheckboxColumn(
                "Must appear in text",
                default=True,
                help="With AI models, one model's value is accepted only if it is printed in the PDF text. "
                "Turn off for values the AI classifies, such as a document type.",
            ),
        },
    )
    return edited.to_dict("records")


def _options(choices: AppChoices) -> None:
    with st.expander("Page-splitting options"):
        if choices.splitting == "rules":
            choices.fallback = st.radio(
                "When the rules can't tell whether a page starts a new document",
                ["new_document", "same_document"],
                format_func=lambda value: "Start a new document (safer: nothing gets merged)"
                if value == "new_document"
                else "Keep it with the previous document",
            )
        choices.blank_pages = st.radio(
            "Blank pages",
            ["ignore", "separator"],
            format_func=lambda value: "Skip them" if value == "ignore" else "Treat them as separators between documents",
            help="Use separators only when blank sheets are placed between documents, not for duplex scans.",
        )


def _cost_box(run_usage: ModelUsage, session_usage: ModelUsage, prices: dict[str, Price]) -> None:
    run, session = estimate(run_usage, prices), estimate(session_usage, prices)
    with st.container(border=True, key="cost_box"):
        left, right = st.columns(2)
        left.metric("Estimated cost of this run", _total(run))
        right.metric("This session so far", _total(session))
        unpriced = [cost for cost in run if cost.usd is None]
        if unpriced:
            st.caption(
                "; ".join(
                    f"{cost.model_id}: {cost.input_tokens + cost.output_tokens:,} tokens, price not set in config.yaml"
                    for cost in unpriced
                )
                + ". The total leaves these out."
            )
        st.caption(
            "Estimate at the list prices in config.yaml. Free tiers and cached-token or batch discounts are not "
            "included; your provider's billing page shows the exact charge."
        )
        if run:
            with st.expander("Cost by model"):
                rows = [
                    {
                        "Model": f"{cost.provider}/{cost.model_id}",
                        "Requests": cost.calls,
                        "Input tokens": cost.input_tokens,
                        "Output tokens": cost.output_tokens,
                        "Cost": "price not set" if cost.usd is None else format_usd(cost.usd),
                    }
                    for cost in run
                ]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _total(costs) -> str:
    """Sum of the priced models; "+" when some models have no price."""
    priced = [cost.usd for cost in costs if cost.usd is not None]
    if costs and not priced:
        return "Price not set"
    return format_usd(sum(priced)) + ("+" if len(priced) < len(costs) else "")


def _test_models(config, api_keys: dict[str, str], config_error: str | None) -> None:
    if config is None:
        st.error(f"Fix the settings first: {config_error}")
        return
    with st.spinner("Sending one tiny request per model…"):
        results = check_models(config, api_keys)
    for name, ok, message in results:
        label = name.replace("_", " ").capitalize()
        if ok:
            st.success(f"{label}: {message}")
        else:
            st.error(f"{label}: {message}")


def _results(result: RunResult, rules_only: bool, workspace: Path, prices: dict[str, Price]) -> None:
    st.subheader("Results")
    if result.stopped:
        st.warning(
            f"The run stopped early ({result.stopped}). The files it had finished are here; "
            "the rest were not processed."
        )
    if rules_only:
        st.warning(
            "These results were produced **without an AI model** and may be less accurate. Check them before use.",
            icon="⚠️",
        )
    for warning in result.warnings:
        st.warning(warning)
    summary = result.summary
    columns = st.columns(5)
    columns[0].metric("Documents found", summary["documents"])
    columns[1].metric("Pages split by rules", f"{summary['pages_by_rules']} of {summary['pages']}")
    columns[2].metric("AI requests", summary["model_requests"])
    columns[3].metric("Confirmed by PDF text", summary["text_verified"])
    columns[4].metric("Failed files", summary["failed"])
    notes = []
    if summary["pages_guessed"]:
        notes.append(f"{summary['pages_guessed']} page(s) had no clear clue and were guessed")
    if summary["duplicates"]:
        notes.append(f"{summary['duplicates']} document(s) flagged as duplicates")
    if summary["failed_documents"]:
        notes.append(
            f"{summary['failed_documents']} document(s) got no usable model answer and are marked FAILED; "
            "upload the file again to retry them"
        )
    if notes:
        st.caption("; ".join(notes) + ". See the Remarks column and the Pages sheet in the workbook.")
    if not rules_only:
        _cost_box(result.model_usage, st.session_state.get("session_usage", {}), prices)
    if result.rows:
        st.dataframe(pd.DataFrame(result.rows), hide_index=True, width="stretch")
    if result.failures:
        st.error("Some files could not be processed:")
        st.dataframe(pd.DataFrame(result.failures, columns=["File", "Reason"]), hide_index=True, width="stretch")
    left, middle, right = st.columns(3)
    left.download_button(
        "Download Excel (.xlsx)",
        result.workbook,
        file_name="extracted_documents.xlsx",
        mime=XLSX_MIME,
        type="primary",
        width="stretch",
    )
    if result.splits_zip:
        middle.download_button(
            "Download split PDFs (.zip)",
            result.splits_zip,
            file_name="split_documents.zip",
            mime="application/zip",
            width="stretch",
        )
    if right.button("Start over", width="stretch", help="Deletes your uploads and results from the server."):
        remove_workspace(workspace)
        for key in ("result", "rules_only", "workspace"):
            st.session_state.pop(key, None)
        st.rerun()
