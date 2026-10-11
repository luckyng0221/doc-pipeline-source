"""Run uploaded PDFs through the pipeline inside one browser session's folder."""

from __future__ import annotations

import io
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeVar

from ..config import PipelineConfig
from ..errors import PipelineError, RunStopped
from ..jobs import claim_pdf
from ..models import BatchResult
from ..pricing import ModelUsage
from ..processor import FAILED, BatchProcessor, ProcessOutcome, ProgressEvent
from ..providers import create_provider
from ..workbook import write_workbook
from .settings import AppLimits

logger = logging.getLogger(__name__)

WORKSPACE_PREFIX = "session-"
# How often a queued run checks for a free slot, touches its workspace so
# cleanup keeps it, and honours a cancel. Far below any sane workspace TTL.
QUEUE_POLL_SECONDS = 30.0
ERROR_HINTS = {
    "400": "the request was rejected; the model may not support the chosen effort (try Provider default) or input",
    "401": "the API key was rejected",
    "403": "the API key is not allowed to use this model",
    "404": "the model was not found for this key; check the model ID",
    "429": "rate limit or quota reached; wait, check your plan, or lower Documents read at once",
}
T = TypeVar("T")


@dataclass
class Upload:
    name: str
    data: bytes


@dataclass
class RunResult:
    outcomes: list[ProcessOutcome]
    workbook: bytes
    splits_zip: bytes | None
    summary: dict[str, int]
    rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    # Tokens and calls per (provider, model ID) of the requests this run made.
    model_usage: ModelUsage = field(default_factory=dict)
    # Why the run stopped before every file was processed, if it did.
    stopped: str | None = None

    @property
    def failures(self) -> list[tuple[str, str]]:
        return [(outcome.source_name, outcome.error or "") for outcome in self.outcomes if outcome.error]


class RunProgress:
    """How far a run has got, shared by the worker thread and the page.

    The run reports through start_file(), event() and finish(); the page
    polls snapshot(), so the elapsed time keeps moving during a slow request.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._started = clock()
        self._file_index, self._file_total, self._file_name = 0, 0, ""
        self._stage = "starting"
        self._done = self._total = 0
        self._extract_started: float | None = None
        self._last_reply: float | None = None
        self._note = ""

    def start_file(self, index: int, total: int, name: str) -> None:
        with self._lock:
            self._file_index, self._file_total, self._file_name = index, total, name
            self._stage, self._done, self._total = "reading", 0, 0
            self._extract_started = None
            self._note = ""

    def event(self, event: ProgressEvent) -> None:
        with self._lock:
            now = self._clock()
            if event.stage == "reply":
                self._last_reply, self._note = now, ""
            elif event.stage == "retry":
                self._note = event.note
            else:
                self._stage = event.stage
                if event.stage == "extracting":
                    self._done, self._total = event.done, event.total
                    if self._extract_started is None:
                        self._extract_started = now

    def finish(self) -> None:
        with self._lock:
            self._stage, self._note = "writing", ""

    def wait(self) -> None:
        """The run is queued until another run on the server finishes."""
        with self._lock:
            self._stage = "waiting"

    def snapshot(self) -> tuple[float, str, str]:
        """Return (fraction done, status line, latest retry note or "")."""
        with self._lock:
            now = self._clock()
            if self._stage == "writing":
                return 1.0, "Writing the workbook…", ""
            if self._stage == "waiting":
                elapsed = _minutes(now - self._started)
                return 0.0, f"Waiting for a free slot: other runs are in progress · {elapsed} elapsed", ""
            name = self._file_name
            if self._file_total > 1:
                name += f" (file {self._file_index + 1} of {self._file_total})"
            # Each file: 10% reading and splitting, 90% extracting fields.
            within = {"reading": 0.0, "splitting": 0.05}.get(self._stage, 0.1)
            if self._stage == "extracting":
                within = 0.1 + 0.9 * (self._done / self._total if self._total else 1.0)
                text = f"Extracting fields from {name}: {self._done} of {self._total} documents done"
            elif self._stage == "splitting":
                text = f"Finding where the documents in {name} start"
            elif self._stage == "reading":
                text = f"Reading the pages of {name}"
            else:
                text = "Starting"
            fraction = (self._file_index + within) / self._file_total if self._file_total else 0.0
            parts = [text, f"{_minutes(now - self._started)} elapsed"]
            if self._stage == "extracting" and self._extract_started is not None and 2 <= self._done < self._total:
                left = (now - self._extract_started) / self._done * (self._total - self._done)
                parts.append(f"about {round(left / 60)} min left" if left >= 90 else "less than 2 min left")
            if self._last_reply is not None:
                parts.append(f"last model reply {now - self._last_reply:.0f} s ago")
            return min(fraction, 1.0), " · ".join(parts), self._note


def _minutes(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


class Refusal(str):
    """A refusal message that also carries a fixed code for the audit log."""

    code: str

    def __new__(cls, message: str, code: str) -> Refusal:
        refusal = super().__new__(cls, message)
        refusal.code = code
        return refusal


class RunSlots:
    """Runs across all sessions of the server.

    `running` runs work at once and up to `waiting` more queue for a slot.
    Each user (a signed-in account, or a session when anonymous) has at most
    one run, so no one can fill the queue alone.
    """

    def __init__(self, running: int, waiting: int) -> None:
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(running)
        self._capacity = running + waiting
        self._users: set[str] = set()

    def admit(self, user: str) -> Refusal | None:
        """Hold a place for the user's run; return why not when there is none."""
        with self._lock:
            if user in self._users:
                return Refusal(
                    "You already have a run in progress. Wait for it to finish, then start the next one.",
                    "user_active",
                )
            if len(self._users) >= self._capacity:
                return Refusal(
                    "The server is busy with other people's runs. Please try again in a few minutes.",
                    "server_busy",
                )
            self._users.add(user)
        return None

    def run(
        self,
        user: str,
        work: Callable[[], T],
        on_wait: Callable[[], None] | None = None,
        keep_alive: Callable[[], None] | None = None,
        abort: Callable[[], bool] | None = None,
    ) -> T | None:
        """Wait for a free slot, do the work, and give up the user's place either way.

        While queued, keep_alive marks the session's workspace as in use on
        every poll, so cleanup never deletes it, and abort ends the wait when
        the run was cancelled while queued (returning None, without work).
        """
        try:
            if not self._slots.acquire(blocking=False):
                if on_wait:
                    on_wait()
                while not self._slots.acquire(timeout=QUEUE_POLL_SECONDS):
                    if keep_alive:
                        keep_alive()
                    if abort and abort():
                        return None
                if keep_alive:
                    keep_alive()
            try:
                return work()
            finally:
                self._slots.release()
        finally:
            self.leave(user)

    def leave(self, user: str) -> None:
        with self._lock:
            self._users.discard(user)


class RunQuota:
    """How many runs each user may start per hour and per day, across the server.

    A sliding-window count of run starts per key: an anonymous visitor's IP
    address, or a signed-in account. In-memory like `RunSlots`, so a restart
    resets it. Only starting a run is charged; a refused one is not.
    """

    HOUR_SECONDS = 3600
    DAY_SECONDS = 86400
    # How often admit() forgets every key whose window has fully expired, so
    # addresses that never come back cannot grow the counter without bound.
    PURGE_INTERVAL_SECONDS = 900

    def __init__(self, per_hour: int, per_day: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._per_hour = per_hour
        self._per_day = per_day
        self._clock = clock
        self._lock = threading.Lock()
        self._starts: dict[str, list[float]] = {}
        self._last_purge = clock()

    def admit(self, key: str) -> Refusal | None:
        """Charge the key one run start, or return why not."""
        now = self._clock()
        with self._lock:
            self._purge(now)
            day = [start for start in self._starts.get(key, ()) if now - start < self.DAY_SECONDS]
            if day:
                self._starts[key] = day
            else:
                self._starts.pop(key, None)  # fully expired: forget the key
            if len(day) >= self._per_day:
                return self._refusal(day, self.DAY_SECONDS, now, "day_quota", self._per_day, "day")
            hour = [start for start in day if now - start < self.HOUR_SECONDS]
            if len(hour) >= self._per_hour:
                return self._refusal(hour, self.HOUR_SECONDS, now, "hour_quota", self._per_hour, "hour")
            day.append(now)
            self._starts[key] = day
            return None

    def _purge(self, now: float) -> None:
        """Drop every key whose newest start left the day window, at most every interval."""
        if now - self._last_purge < self.PURGE_INTERVAL_SECONDS:
            return
        self._last_purge = now
        for key, starts in list(self._starts.items()):
            if now - starts[-1] >= self.DAY_SECONDS:
                del self._starts[key]

    def _refusal(
        self, window: list[float], span: float, now: float, code: str, limit: int, noun: str
    ) -> Refusal:
        # The oldest start in the window is the first to leave it, so waiting
        # until then is exactly when the next place frees up.
        minutes = max(1, math.ceil((window[0] + span - now) / 60))
        return Refusal(
            f"You have already started {limit} runs in the last {noun}. Please try again in {minutes} minutes.",
            code,
        )


class ModelTestQuota:
    """How often each visitor may test models per sliding hour, across the server.

    Testing models sends one tiny request per configured model, and unlike a
    run it needs no uploads, so it gets its own hourly cap, counted per key
    like the run quota and sharing nothing with it. In-memory, so a restart
    resets it.
    """

    HOUR_SECONDS = 3600
    PURGE_INTERVAL_SECONDS = 900

    def __init__(self, per_hour: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._per_hour = per_hour
        self._clock = clock
        self._lock = threading.Lock()
        self._tests: dict[str, list[float]] = {}
        self._last_purge = clock()

    def admit(self, key: str) -> Refusal | None:
        """Charge the key one model test, or return why not."""
        now = self._clock()
        with self._lock:
            self._purge(now)
            window = [start for start in self._tests.get(key, ()) if now - start < self.HOUR_SECONDS]
            if window:
                self._tests[key] = window
            else:
                self._tests.pop(key, None)  # fully expired: forget the key
            if len(window) >= self._per_hour:
                minutes = max(1, math.ceil((window[0] + self.HOUR_SECONDS - now) / 60))
                return Refusal(
                    f"You have already tested models {self._per_hour} times in the last hour. "
                    f"Please try again in {minutes} minutes.",
                    "test_quota",
                )
            window.append(now)
            self._tests[key] = window
            return None

    def _purge(self, now: float) -> None:
        """Drop every key whose newest test left the hour window, at most every interval."""
        if now - self._last_purge < self.PURGE_INTERVAL_SECONDS:
            return
        self._last_purge = now
        for key, starts in list(self._tests.items()):
            if now - starts[-1] >= self.HOUR_SECONDS:
                del self._tests[key]


class RunControl:
    """Stops a run: cancelled by the user, or out of time.

    The run stops before its next model request or file; a request already in
    flight finishes first.

    The deadline starts only when the run leaves the queue (start()), so the
    wait for a free slot does not eat into the run's time budget.
    """

    def __init__(
        self,
        timeout_minutes: int | None = None,
        cancelled: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        # Set when the session's folder was deleted under the run (sign-out),
        # so the run can leave nothing behind at its path.
        self.workspace_removed = threading.Event()
        self.timeout_minutes = timeout_minutes
        self._clock = clock
        self.deadline: float | None = None

    def start(self) -> None:
        """Begin the time budget; called once the run is about to work."""
        if self.timeout_minutes is not None:
            self.deadline = self._clock() + self.timeout_minutes * 60

    def check(self) -> None:
        """Raise RunStopped when the run should stop (static reasons only)."""
        if self.cancelled.is_set():
            raise RunStopped("the run was cancelled")
        if self.deadline is not None and self._clock() >= self.deadline:
            raise RunStopped(f"the run passed its {self.timeout_minutes}-minute time limit")


def workspace_root() -> Path:
    """Where session folders live; DOC_PIPELINE_WORKSPACES overrides the temp folder."""
    configured = os.environ.get("DOC_PIPELINE_WORKSPACES", "").strip()
    if configured:
        return Path(configured)
    root = Path(tempfile.gettempdir()) / "doc-pipeline-app"
    root.mkdir(mode=0o700, exist_ok=True)
    _check_private(root)
    return root


def _check_private(root: Path) -> None:
    """Refuse a shared-temp folder that another user made or linked (POSIX).

    Its owner could delete or swap the session folders inside it.
    """
    if os.name != "posix":
        return
    info = root.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
        raise PermissionError(f"The workspace folder {root} is a link or belongs to another user.")


def new_workspace(root: Path | None = None) -> Path:
    root = root or workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=WORKSPACE_PREFIX, dir=root))


def reserve_workspace(root: Path | None = None) -> Path:
    """A fresh workspace path without creating anything; ensure_workspace makes it.

    A page load that never uploads, tests models, or starts a run then leaves
    no folder behind, so scripted visits cannot pile them up.
    """
    root = root or workspace_root()
    return root / f"{WORKSPACE_PREFIX}{uuid.uuid4().hex}"


def ensure_workspace(path: Path) -> None:
    """Create the session folder (and nothing else) once a run needs it.

    The root already exists by then: reserve_workspace made it through
    workspace_root(), which creates it private (mode 0700) and refuses one it
    does not own, so mkdir here only ever adds the folder itself.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)


def touch_workspace(workspace: Path) -> None:
    """Mark a session folder as in use so cleanup keeps it."""
    try:
        os.utime(workspace)
    except OSError:
        pass


def remove_workspace(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)


def cleanup_workspaces(ttl_minutes: int, root: Path | None = None, now: float | None = None) -> int:
    """Delete session folders untouched for longer than ttl_minutes."""
    root = root or workspace_root()
    if not root.is_dir():
        return 0
    cutoff = (now or time.time()) - ttl_minutes * 60
    removed = 0
    for entry in root.iterdir():
        try:
            stale = entry.is_dir() and entry.name.startswith(WORKSPACE_PREFIX) and entry.stat().st_mtime < cutoff
        except OSError:
            continue
        if stale:
            remove_workspace(entry)
            removed += 1
    return removed


# How often the background sweeper removes expired session folders.
CLEANUP_INTERVAL_SECONDS = 15 * 60


def sweep_workspaces(ttl_minutes: int, stop: threading.Event, interval_seconds: float = CLEANUP_INTERVAL_SECONDS) -> None:
    """Delete expired session folders every interval, until stop is set.

    Runs in a daemon thread, so cleanup does not depend on page loads. One
    failed sweep is logged and skipped, so the thread outlives it.
    """
    while not stop.wait(interval_seconds):
        try:
            cleanup_workspaces(ttl_minutes)
        except Exception as exc:
            logger.error("Workspace sweep failed (%s).", type(exc).__name__)


def check_uploads(uploads: list[Upload], limits: AppLimits) -> list[str]:
    problems = []
    if len(uploads) > limits.max_files:
        problems.append(f"Upload at most {limits.max_files} PDFs per run ({len(uploads)} selected).")
    for upload in uploads:
        if len(upload.data) > limits.max_file_mb * 1024 * 1024:
            problems.append(f"{upload.name} is larger than {limits.max_file_mb} MB.")
    return problems


def upload_refusal_codes(uploads: list[Upload], limits: AppLimits) -> list[str]:
    """The same checks as check_uploads, as the audit log's fixed refusal codes.

    The messages name files, so the page shows those and the log gets codes.
    """
    codes = []
    if len(uploads) > limits.max_files:
        codes.append("too_many_files")
    if any(len(upload.data) > limits.max_file_mb * 1024 * 1024 for upload in uploads):
        codes.append("file_too_large")
    return codes


def run_uploads(
    config: PipelineConfig,
    api_keys: dict[str, str],
    uploads: list[Upload],
    workspace: Path,
    progress: RunProgress | None = None,
    control: RunControl | None = None,
) -> RunResult:
    """Process each upload with the pipeline's duplicate and error handling.

    The run stops before its next model request or file (time limit or cancel;
    a request in flight finishes first): files it finished are kept, and the
    file it was on and the ones it never reached are reported as failed with
    the reason. A folder deleted while the run works
    (sign-out) stops it too, and nothing is rebuilt at its path.
    """
    control = control or RunControl()
    control.start()
    touch_workspace(workspace)
    if not workspace.is_dir():
        # Gone before the start (a sign-out while queued, say): constructing
        # the processor would rebuild the whole tree, so give up first.
        raise RunStopped("the run's workspace was removed")
    try:
        return _process_uploads(config, api_keys, uploads, workspace, progress, control)
    finally:
        if control.workspace_removed.is_set():
            # Sign-out deleted the folder under the run: the documents still in
            # flight write their archives and results after the stop, so sweep
            # the path once more when the run ends.
            remove_workspace(workspace)


def _process_uploads(
    config: PipelineConfig,
    api_keys: dict[str, str],
    uploads: list[Upload],
    workspace: Path,
    progress: RunProgress | None,
    control: RunControl,
) -> RunResult:
    """Run the uploads; the workspace exists and the time budget has started."""

    def on_event(event: ProgressEvent) -> None:
        # A long run keeps its folder fresh, so cleanup never deletes it.
        touch_workspace(workspace)
        if progress:
            progress.event(event)

    # An explicit (possibly empty) key map: the app never reads server keys.
    processor = BatchProcessor(config, api_keys=dict(api_keys), progress=on_event, stop_check=control.check)
    staging = workspace / "uploads"
    outcomes: list[ProcessOutcome] = []
    reason: str | None = None
    for index, upload in enumerate(uploads):
        touch_workspace(workspace)
        folder: Path | None = None
        started = False
        try:
            control.check()
            if progress:
                progress.start_file(index, len(uploads), upload.name)
            started = True
            if not workspace.is_dir():
                # Deleted under the run: staging the next file would rebuild it.
                raise RunStopped("the run's workspace was removed")
            folder = staging / uuid.uuid4().hex
            folder.mkdir(parents=True)
            path = folder / _safe_upload_name(upload.name)
            path.write_bytes(upload.data)
            claimed = claim_pdf(path, config.paths.processing, move_source=True)
            outcomes.append(
                processor.handle_claimed_pdf(claimed.path, source_name=claimed.source_name, run_id=claimed.run_id)
            )
        except RunStopped as exc:
            reason = str(exc)
            outcomes.append(
                ProcessOutcome(
                    upload.name,
                    None,
                    None,
                    f"Stopped part-way through ({reason})." if started else f"Not processed: {reason}.",
                    workspace,
                )
            )
            outcomes.extend(
                ProcessOutcome(remaining.name, None, None, f"Not processed: {reason}.", workspace)
                for remaining in uploads[len(outcomes) :]
            )
            break
        finally:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
    if progress:
        progress.finish()
    if control.workspace_removed.is_set() or not workspace.is_dir():
        # Sign-out deleted the folder under the run: a workbook nobody can
        # fetch must not rebuild it.
        raise RunStopped(reason or "the run's workspace was removed")

    batches = [outcome.batch for outcome in outcomes if outcome.batch is not None]
    failures = [(outcome.source_name, outcome.error or "") for outcome in outcomes if outcome.error]
    workbook_path = workspace / "output" / f"results_{uuid.uuid4().hex[:8]}.xlsx"
    write_workbook(batches, config.fields, workbook_path, failures)
    workbook = workbook_path.read_bytes()
    workbook_path.unlink(missing_ok=True)
    usage = processor.usage_by_model()
    return RunResult(
        outcomes=outcomes,
        workbook=workbook,
        splits_zip=_splits_zip(outcomes),
        summary=_summary(batches, failures, usage),
        rows=_preview_rows(batches, config),
        warnings=_warnings(batches, config),
        model_usage=usage,
        stopped=reason,
    )


def check_models(config: PipelineConfig, api_keys: dict[str, str]) -> list[tuple[str, bool, str]]:
    """Send one tiny request per model; return (model name, ok, message)."""
    limits = replace(config.limits, retry_attempts=1)
    results = []
    for model in config.models:
        provider = create_provider(model, limits, api_keys.get(model.name, ""))
        try:
            provider.check_connection()
        except PipelineError as exc:
            results.append((model.name, False, explain_error(str(exc))))
        else:
            results.append((model.name, True, f"{model.provider}/{model.model} answered"))
    return results


def explain_error(message: str) -> str:
    """Add a plain-English hint to a sanitized provider error."""
    status = re.search(r"HTTP (\d{3})", message)
    hint = ERROR_HINTS.get(status.group(1)) if status else None
    if hint is None and "no api key" in message.casefold():
        hint = "enter an API key"
    return f"{message}: {hint}" if hint else message


def _safe_upload_name(name: str) -> str:
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip() or "upload.pdf"
    base = re.sub(r"[^\w .()-]+", "_", base)[:120]
    return base if base.casefold().endswith(".pdf") else f"{base}.pdf"


def _splits_zip(outcomes: list[ProcessOutcome]) -> bytes | None:
    buffer = io.BytesIO()
    count = 0
    used: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for outcome in outcomes:
            splits = outcome.archive_dir / "splits"
            if outcome.batch is None or not splits.is_dir():
                continue
            base = re.sub(r"[^\w.-]+", "_", Path(outcome.source_name).stem)[:60] or "document"
            # Uploads with the same name get scan, scan_2, ... so no file is
            # overwritten on unzip (case-insensitive, as on Windows).
            folder, copy = base, 1
            while folder.casefold() in used:
                copy += 1
                folder = f"{base}_{copy}"
            used.add(folder.casefold())
            for part in sorted(splits.glob("*.pdf")):
                archive.write(part, f"{folder}/{part.name}")
                count += 1
    return buffer.getvalue() if count else None


def _summary(batches: list[BatchResult], failures: list[tuple[str, str]], usage: ModelUsage) -> dict[str, int]:
    pages = [page for batch in batches for page in batch.page_decisions]
    documents = [document for batch in batches for document in batch.documents]
    return {
        "files": len(batches) + len(failures),
        "failed": len(failures),
        "documents": len(documents),
        "pages": len(pages),
        "pages_by_rules": sum(page.get("decided_by") == "rule" for page in pages),
        "pages_by_model": sum(page.get("decided_by") == "model" for page in pages),
        "pages_guessed": sum(page.get("decided_by") == "guess" for page in pages),
        # Requests this run made: reused (cached) files make none.
        "model_requests": sum(tokens["calls"] for tokens in usage.values()),
        "text_verified": sum(document.validation_status == "TEXT_VERIFIED" for document in documents),
        "failed_documents": sum(document.validation_status == FAILED for document in documents),
        "duplicates": sum(document.duplicate_status != "NONE" for document in documents),
    }


def _preview_rows(batches: list[BatchResult], config: PipelineConfig) -> list[dict[str, Any]]:
    rows = []
    for batch in batches:
        for document in batch.documents:
            row: dict[str, Any] = {
                "Source PDF": batch.source_name,
                "Document #": document.index,
                "Pages": f"{document.page_start}-{document.page_end}",
                "Status": document.validation_status,
                "Duplicate": document.duplicate_status,
            }
            # Text only: models may return 12.5 for one document and "12.50" for
            # another, and a mixed column cannot be shown as a table.
            row.update(
                {
                    spec.name: None if document.fields.get(spec.name) is None else str(document.fields.get(spec.name))
                    for spec in config.fields
                }
            )
            row["Remarks"] = " | ".join(document.remarks)
            rows.append(row)
    return rows


def _warnings(batches: list[BatchResult], config: PipelineConfig) -> list[str]:
    warnings = []
    for batch in batches:
        pages = batch.page_decisions
        if pages and all(not page.get("text_chars") for page in pages) and not config.models:
            warnings.append(
                f"{batch.source_name} has no text layer (an image-only scan), so rules-only mode could not read it. "
                "Add an API key, or use a scan with OCR text."
            )
    return warnings
