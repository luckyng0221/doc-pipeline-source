from __future__ import annotations

import io
import logging
import os
import stat
import tempfile
import threading
import time
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import pytest
from openpyxl import load_workbook

from doc_pipeline.app.session import (
    ModelTestQuota,
    RunControl,
    RunProgress,
    RunQuota,
    RunSlots,
    Upload,
    check_uploads,
    cleanup_workspaces,
    ensure_workspace,
    explain_error,
    new_workspace,
    remove_workspace,
    run_uploads,
    sweep_workspaces,
    touch_workspace,
    workspace_root,
)
from doc_pipeline.app.settings import (
    AppChoices,
    AppLimits,
    ModelSlot,
    build_app_config,
    describe_mode,
    load_app_defaults,
    preset_rows,
)
from doc_pipeline.errors import ConfigurationError, RunStopped
from doc_pipeline.processor import ProgressEvent

ROOT = Path(__file__).parents[1]
DEFAULTS = load_app_defaults(ROOT / "config.yaml")
AUTH = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "test-secret",
    "google": {
        "client_id": "id",
        "client_secret": "secret",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    },
}


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

    Every test gets a stand-in that cannot change this process's dumpable
    flag, and a stand-in SDK import that cannot touch this process's loaded
    modules; the once-per-process globals are put back to their "not yet
    done" value so the page's initializers really run (test_hardening.py
    covers the real prctl call, in subprocesses).
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


def _choices(**changes: object) -> AppChoices:
    choices = AppChoices(fields=preset_rows(DEFAULTS), date_order="day_first")
    for name, value in changes.items():
        setattr(choices, name, value)
    return choices


def test_defaults_come_from_config_without_server_keys(monkeypatch) -> None:
    rows = preset_rows(DEFAULTS)

    assert DEFAULTS.limits == AppLimits(
        max_files=5,
        max_file_mb=20,
        max_pages=50,
        workspace_ttl_minutes=120,
        max_concurrent_runs=2,
        max_waiting_runs=4,
        run_timeout_minutes=60,
        max_runs_per_hour=10,
        max_runs_per_day=40,
    )
    assert [row["Field"] for row in rows if row["Include"]] == [
        "Document Number",
        "Document Date",
        "Vendor Name",
        "Customer Name",
        "Currency",
        "Tax / SST Amount",
        "Total Amount",
    ]
    assert "models" not in DEFAULTS.base_raw and "boundary_model" not in DEFAULTS.base_raw


def test_app_limit_values_are_checked(tmp_path: Path) -> None:
    def limits(app_section: str) -> AppLimits:
        path = tmp_path / "config.yaml"
        path.write_text(f"app:\n{app_section}", encoding="utf-8")
        return load_app_defaults(path).limits

    assert limits("  run_timeout_minutes: 90\n").run_timeout_minutes == 90
    assert limits("  {}\n").run_timeout_minutes == 60  # absent: the default
    with pytest.raises(ConfigurationError, match="whole numbers"):
        limits("  run_timeout_minutes: soon\n")
    with pytest.raises(ConfigurationError, match="at least 1"):
        limits("  run_timeout_minutes: 0\n")
    # A run must outlive its folder: the TTL is checked against the time limit.
    with pytest.raises(ConfigurationError, match="longer than"):
        limits("  run_timeout_minutes: 999\n")


def test_run_quota_limits_come_from_config_and_are_checked(tmp_path: Path) -> None:
    def limits(app_section: str) -> AppLimits:
        path = tmp_path / "config.yaml"
        path.write_text(f"app:\n{app_section}", encoding="utf-8")
        return load_app_defaults(path).limits

    assert limits("  {}\n").max_runs_per_hour == 10  # absent: the default
    assert limits("  {}\n").max_runs_per_day == 40
    parsed = limits("  max_runs_per_hour: 3\n  max_runs_per_day: 9\n")
    assert (parsed.max_runs_per_hour, parsed.max_runs_per_day) == (3, 9)
    with pytest.raises(ConfigurationError, match="whole numbers"):
        limits("  max_runs_per_hour: soon\n")
    with pytest.raises(ConfigurationError, match="at least 1"):
        limits("  max_runs_per_day: 0\n")
    with pytest.raises(ConfigurationError, match="app.max_runs_per_hour.*app.max_runs_per_day"):
        limits("  max_runs_per_hour: 50\n  max_runs_per_day: 40\n")


def test_model_test_quota_limits_come_from_config_and_are_checked(tmp_path: Path) -> None:
    def limits(app_section: str) -> AppLimits:
        path = tmp_path / "config.yaml"
        path.write_text(f"app:\n{app_section}", encoding="utf-8")
        return load_app_defaults(path).limits

    assert limits("  {}\n").max_model_tests_per_hour == 20  # absent: the default
    assert limits("  max_model_tests_per_hour: 7\n").max_model_tests_per_hour == 7
    with pytest.raises(ConfigurationError, match="whole numbers"):
        limits("  max_model_tests_per_hour: soon\n")
    with pytest.raises(ConfigurationError, match="app.max_model_tests_per_hour must be at least 1"):
        limits("  max_model_tests_per_hour: 0\n")


def test_model_slots_become_roles_and_keys(tmp_path: Path) -> None:
    slots = [
        ModelSlot("gemini", " key-1 ", "gemini-3.8-flash", "low"),
        ModelSlot("openai", "key-2", "gpt-4o-mini"),
        ModelSlot("glm", "key-3", "glm-4.6v-flash", "off", "https://api.z.ai/api/paas/v4/"),
    ]

    config, keys = build_app_config(DEFAULTS, _choices(slots=slots, splitting=1), tmp_path)

    assert [(model.name, model.role, model.input) for model in config.models] == [
        ("model_1", "primary", "pdf"),
        ("model_2", "primary", "images"),
        ("model_3", "arbiter", "images"),
    ]
    assert config.models[0].effort == "low"
    assert config.models[2].base_url == "https://api.z.ai/api/paas/v4/"
    assert config.boundary_model.name == "model_1"
    assert keys == {"model_1": "key-1", "model_2": "key-2", "model_3": "key-3"}
    assert DEFAULTS.extraction_workers == 3 and config.limits.extraction_workers == 3
    assert build_app_config(DEFAULTS, _choices(workers=6), tmp_path)[0].limits.extraction_workers == 6
    assert config.paths.registry == tmp_path / "registry.db"
    assert config.fingerprint_fields == ("Document Number", "Document Date", "Vendor Name", "Total Amount")
    assert "Model 1 Gemini (Google) gemini-3.8-flash (runs first, effort low)" in describe_mode(
        _choices(slots=slots, splitting=1)
    )


def test_invalid_choices_explain_what_to_fix(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="enter an API key"):
        build_app_config(DEFAULTS, _choices(slots=[ModelSlot("gemini", "", "gemini-3.8-flash")]), tmp_path)
    with pytest.raises(ConfigurationError, match="in order"):
        build_app_config(DEFAULTS, _choices(slots=[ModelSlot(), ModelSlot("openai", "k", "gpt-4o-mini")]), tmp_path)
    with pytest.raises(ConfigurationError, match="at least one field"):
        build_app_config(DEFAULTS, _choices(fields=[{**preset_rows(DEFAULTS)[0], "Include": False}]), tmp_path)
    assert check_uploads([Upload("big.pdf", b"x" * (2 * 1024 * 1024))], AppLimits(max_file_mb=1)) == [
        "big.pdf is larger than 1 MB."
    ]
    assert explain_error("model_1: AuthenticationError (HTTP 401)").endswith("the API key was rejected")


def test_rules_only_run_splits_extracts_and_flags_duplicates(tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    batch = _pdf(
        [
            "TAX INVOICE\nInvoice No: A-1001\nDate: 24/02/2025\nPage 1 of 2",
            "TAX INVOICE\nPage 2 of 2\nTotal Amount: 12.50",
            "OFFICIAL RECEIPT\nReceipt No: R-77\nDate: 25/02/2025\nTotal 12.50",
        ]
    )
    uploads = [Upload("batch.pdf", batch), Upload("again.pdf", batch), Upload("broken.pdf", b"not a pdf")]

    result = run_uploads(config, keys, uploads, workspace)

    assert keys == {}
    assert result.model_usage == {} and result.summary["model_requests"] == 0
    assert result.summary["documents"] == 4
    assert result.summary["failed"] == 1
    assert result.failures == [("broken.pdf", "The file is not a readable PDF.")]
    # A failed open must not leave the file locked (Windows), or it never reaches errors/.
    broken = next(outcome for outcome in result.outcomes if outcome.error)
    assert (broken.archive_dir / "failure.txt").is_file()
    first = [row for row in result.rows if row["Source PDF"] == "batch.pdf"]
    assert [(row["Pages"], row["Status"], row["Document Number"]) for row in first] == [
        ("1-2", "RULES_ONLY", "A-1001"),
        ("3-3", "RULES_ONLY", "R-77"),
    ]
    assert first[0]["Total Amount"] == "12.50" and first[0]["Document Date"] == "2025-02-24"
    assert {row["Duplicate"] for row in result.rows if row["Source PDF"] == "again.pdf"} == {"DUPLICATE"}
    workbook = load_workbook(io.BytesIO(result.workbook))
    assert {"Documents", "Pages", "Failed files", "Run info"} <= set(workbook.sheetnames)
    names = zipfile.ZipFile(io.BytesIO(result.splits_zip)).namelist()
    assert "batch/document_001_pages_0001-0002.pdf" in names


def test_uploads_with_the_same_name_get_separate_zip_folders(tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    uploads = [Upload("scan.pdf", _pdf(["INVOICE\nInvoice No: A-1"])), Upload("Scan.pdf", _pdf(["INVOICE\nInvoice No: Z-9"]))]

    result = run_uploads(config, keys, uploads, workspace)

    names = zipfile.ZipFile(io.BytesIO(result.splits_zip)).namelist()
    assert names == ["scan/document_001_pages_0001-0001.pdf", "Scan_2/document_001_pages_0001-0001.pdf"]


class _StopAfterFirstFile:
    """Progress stub that acts once the first file has finished.

    With a workspace, it also ages the folder while the first file is worked
    on, so the folder only stays fresh if the run touches it again at the
    next file boundary.
    """

    def __init__(self, action, workspace: Path | None = None) -> None:
        self.action = action
        self.workspace = workspace
        self.files: list[int] = []
        self.current = -1

    def start_file(self, index: int, total: int, name: str) -> None:
        self.files.append(index)
        self.current = index
        if index == 1:
            self.action()

    def event(self, event: ProgressEvent) -> None:
        if self.workspace is not None and self.current == 0:
            hours_ago = time.time() - 7200
            os.utime(self.workspace, (hours_ago, hours_ago))

    def finish(self) -> None:
        pass


def test_a_run_over_its_time_limit_keeps_finished_files_and_its_folder(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.registry import Registry

    # The registry writes next to the folder (db, -wal, -shm) and freshens it
    # the way a touch does, so this run records nothing there: the touches
    # must keep the folder alive on their own.
    monkeypatch.setattr(Registry, "save_batch", lambda self, **_: None)
    monkeypatch.setattr(Registry, "find_possible_duplicate", lambda self, *_, **__: None)
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    uploads = [Upload("first.pdf", _pdf(["TAX INVOICE\nInvoice No: A-1001\nDate: 24/02/2025\nTotal: 12.50"]))]
    uploads += [Upload("second.pdf", _pdf(["TAX INVOICE\nInvoice No: B-2002\nDate: 25/02/2025\nTotal: 20.00"]))]
    uploads += [Upload("third.pdf", _pdf(["TAX INVOICE\nInvoice No: C-3003\nDate: 26/02/2025\nTotal: 30.00"]))]
    now = [0.0]
    control = RunControl(timeout_minutes=1, clock=lambda: now[0])

    def over_limit() -> None:
        now[0] = 3600  # the deadline passes during file 2
        # Mid-run, with the folder stale from file 1: the boundary touch for
        # this file keeps it, so cleanup cannot take an in-progress folder.
        assert cleanup_workspaces(60, root=tmp_path) == 0
        assert workspace.exists()

    progress = _StopAfterFirstFile(over_limit, workspace)

    result = run_uploads(config, keys, uploads, workspace, progress, control)

    assert progress.files == [0, 1]
    assert result.stopped == "the run passed its 1-minute time limit"
    assert result.failures == [
        ("second.pdf", "Stopped part-way through (the run passed its 1-minute time limit)."),
        ("third.pdf", "Not processed: the run passed its 1-minute time limit."),
    ]
    assert [row["Source PDF"] for row in result.rows] == ["first.pdf"]
    workbook = load_workbook(io.BytesIO(result.workbook))
    assert any("first.pdf" in str(cell) for row in workbook["Documents"].iter_rows(values_only=True) for cell in row)
    assert cleanup_workspaces(60, root=tmp_path) == 0
    assert workspace.exists()


def test_cancelling_mid_run_reports_the_rest_as_not_processed(tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    uploads = [Upload("first.pdf", _pdf(["TAX INVOICE\nInvoice No: A-1001\nDate: 24/02/2025\nTotal: 12.50"]))]
    uploads += [Upload("second.pdf", _pdf(["TAX INVOICE\nInvoice No: B-2002\nDate: 25/02/2025\nTotal: 20.00"]))]
    uploads += [Upload("third.pdf", _pdf(["TAX INVOICE\nInvoice No: C-3003\nDate: 26/02/2025\nTotal: 30.00"]))]
    control = RunControl()
    progress = _StopAfterFirstFile(control.cancelled.set)

    result = run_uploads(config, keys, uploads, workspace, progress, control)

    assert result.stopped == "the run was cancelled"
    assert result.failures == [
        ("second.pdf", "Stopped part-way through (the run was cancelled)."),
        ("third.pdf", "Not processed: the run was cancelled."),
    ]
    assert [row["Source PDF"] for row in result.rows] == ["first.pdf"]


def test_a_run_cancelled_before_it_starts_fails_every_file(tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    uploads = [Upload("first.pdf", _pdf(["INVOICE\nInvoice No: A-1"])), Upload("second.pdf", _pdf(["INVOICE\nInvoice No: B-2"]))]
    control = RunControl()
    control.cancelled.set()

    result = run_uploads(config, keys, uploads, workspace, control=control)

    assert result.stopped == "the run was cancelled"
    # No file was started, so none is reported as stopped part-way through.
    assert result.failures == [
        ("first.pdf", "Not processed: the run was cancelled."),
        ("second.pdf", "Not processed: the run was cancelled."),
    ]
    assert result.rows == [] and isinstance(result.workbook, bytes)


def test_a_run_cancelled_while_queued_never_recreates_its_workspace(tmp_path: Path, monkeypatch) -> None:
    import doc_pipeline.processor as processor_module

    from doc_pipeline.models import ModelConfig

    created: list[str] = []

    def no_provider(model: ModelConfig, limits, api_key=None):
        created.append(model.name)
        raise AssertionError("a cancelled run must not create providers")

    monkeypatch.setattr(processor_module, "create_provider", no_provider)
    workspace = new_workspace(tmp_path)
    choices = _choices(slots=[ModelSlot("gemini", "key", "test-model")], splitting=1)
    config, keys = build_app_config(DEFAULTS, choices, workspace)
    uploads = [Upload("first.pdf", _pdf(["INVOICE\nInvoice No: A-1"]))]
    control = RunControl()
    control.cancelled.set()
    remove_workspace(workspace)

    with pytest.raises(RunStopped, match="workspace was removed"):
        run_uploads(config, keys, uploads, workspace, control=control)

    assert not created  # not one model request
    assert not workspace.exists() and list(tmp_path.iterdir()) == []


def test_a_workspace_removed_mid_run_is_not_rebuilt(tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    uploads = [Upload("first.pdf", _pdf(["INVOICE\nInvoice No: A-1"])), Upload("second.pdf", _pdf(["INVOICE\nInvoice No: B-2"]))]
    control = RunControl()

    def sign_out_mid_run() -> None:
        # What _sign_out does: stop the job, then delete its folder.
        control.cancelled.set()
        control.workspace_removed.set()
        remove_workspace(workspace)

    progress = _StopAfterFirstFile(sign_out_mid_run)

    with pytest.raises(RunStopped, match="workspace was removed"):
        run_uploads(config, keys, uploads, workspace, progress, control)

    # The staging, processing, and workbook folders were not rebuilt.
    assert not workspace.exists() and list(tmp_path.iterdir()) == []


def test_a_sign_out_during_a_model_request_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    import doc_pipeline.processor as processor_module

    from doc_pipeline.models import ModelConfig
    from doc_pipeline.providers import ProviderResponse

    USAGE = {"prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0, "cached_tokens": 0, "total_tokens": 0}

    class BlockedProvider:
        """One model request that returns only when the test releases it."""

        def __init__(self) -> None:
            self.calls = 0
            self.release = threading.Event()
            self.config = ModelConfig(name="model_1", provider="gemini", model="test-model", api_key_env="X_API_KEY")

        def generate_json(self, prompt: str, parts) -> ProviderResponse:
            self.calls += 1
            assert self.release.wait(30), "the model request was never released"
            return ProviderResponse(data={"fields": {}}, usage=USAGE)

    provider = BlockedProvider()
    monkeypatch.setattr(processor_module, "create_provider", lambda model, limits, api_key=None: provider)
    workspace = new_workspace(tmp_path)
    choices = _choices(slots=[ModelSlot("gemini", "key", "test-model")], splitting=1)
    config, keys = build_app_config(DEFAULTS, choices, workspace)
    # Marked pages: the rules settle the boundaries, so the only request is
    # the extraction of the single document.
    uploads = [Upload("batch.pdf", _pdf(["Acme Trading\nPage 1 of 2\nTotal due 12.50", "Acme Trading details\nPage 2 of 2"]))]
    control = RunControl()
    outcomes: list[BaseException] = []

    def work() -> None:
        try:
            run_uploads(config, keys, uploads, workspace, RunProgress(), control)
        except BaseException as exc:  # the job thread reports the stop to nobody
            outcomes.append(exc)

    job = threading.Thread(target=work, name="doc-pipeline-run")

    job.start()
    try:
        deadline = time.time() + 10
        while provider.calls == 0 and time.time() < deadline:
            time.sleep(0.005)
        assert provider.calls == 1  # the request is in flight

        # What _sign_out does, mid-request: stop the job, mark the folder as
        # removed, delete it.
        control.cancelled.set()
        control.workspace_removed.set()
        remove_workspace(workspace)
        provider.release.set()
    finally:
        job.join(30)

    assert not job.is_alive()
    assert provider.calls == 1  # the in-flight request finished; no new one started
    assert [type(exc) for exc in outcomes] == [RunStopped]
    # Nothing the run wrote after the sign-out was left at the folder's path.
    assert not workspace.exists() and list(tmp_path.iterdir()) == []


def test_old_session_folders_are_deleted(tmp_path: Path) -> None:
    old, fresh = new_workspace(tmp_path), new_workspace(tmp_path)
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))

    assert cleanup_workspaces(60, root=tmp_path) == 1
    assert not old.exists() and fresh.exists()


def test_the_background_sweep_deletes_old_folders_without_a_page_load(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOC_PIPELINE_WORKSPACES", str(tmp_path))
    old, fresh = new_workspace(tmp_path), new_workspace(tmp_path)
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    stop = threading.Event()
    sweeper = threading.Thread(target=sweep_workspaces, args=(60, stop, 0.01), daemon=True)

    sweeper.start()
    try:
        deadline = time.time() + 5
        while old.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert not old.exists() and fresh.exists()
    finally:
        stop.set()
        sweeper.join(5)

    assert not sweeper.is_alive()  # the loop ends when its stop event is set


def test_the_sweeper_survives_a_failed_sweep_and_keeps_its_log_clean(tmp_path: Path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import session as session_module

    monkeypatch.setenv("DOC_PIPELINE_WORKSPACES", str(tmp_path))
    old, fresh = new_workspace(tmp_path), new_workspace(tmp_path)
    two_hours_ago = time.time() - 7200
    os.utime(old, (two_hours_ago, two_hours_ago))
    real_cleanup = session_module.cleanup_workspaces
    sweeps = [0]

    def failing_once(ttl_minutes: int, root: Path | None = None, now: float | None = None) -> int:
        sweeps[0] += 1
        if sweeps[0] == 1:
            raise OSError(f"cannot sweep {root}")  # deliberately carries a path
        return real_cleanup(ttl_minutes, root=root, now=now)

    monkeypatch.setattr(session_module, "cleanup_workspaces", failing_once)
    stop = threading.Event()
    sweeper = threading.Thread(target=sweep_workspaces, args=(60, stop, 0.01), daemon=True)

    with caplog.at_level(logging.ERROR):
        sweeper.start()
        try:
            deadline = time.time() + 5
            while old.exists() and time.time() < deadline:
                time.sleep(0.005)
            assert not old.exists() and fresh.exists()  # the next sweep still worked
        finally:
            stop.set()
            sweeper.join(5)

    assert not sweeper.is_alive()
    # The only error is the fixed message plus the exception class: no path.
    assert [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR] == [
        "Workspace sweep failed (OSError)."
    ]


def test_the_page_starts_one_background_cleanup_thread(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert any(thread.name == "doc-pipeline-cleanup" and thread.daemon for thread in threading.enumerate())


def test_the_page_hardens_the_server_process_once_per_run(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    fired: list[bool] = []

    def recorder() -> bool:
        fired.append(True)
        return True

    monkeypatch.setattr(ui, "set_non_dumpable", recorder)
    monkeypatch.setattr(ui, "_hardened", None)

    app = _app(tmp_path, monkeypatch).run()
    app.run()  # a rerun, as every page load is

    assert not app.exception
    assert fired == [True]  # once per server process, not once per page load


def test_a_failed_hardening_logs_one_constant_warning_on_linux(monkeypatch, caplog) -> None:
    from types import SimpleNamespace

    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "set_non_dumpable", lambda: False)
    monkeypatch.setattr(ui, "_hardened", None)
    monkeypatch.setattr(ui, "sys", SimpleNamespace(platform="linux"))

    with caplog.at_level(logging.WARNING, logger=ui.logger.name):
        assert ui._harden_process() is False
        assert ui._harden_process() is False  # the second page load does not call it again

    assert [record.getMessage() for record in caplog.records if record.name == ui.logger.name] == [
        "Server hardening failed: the process is still dumpable."
    ]


@pytest.mark.parametrize(("platform", "result"), [("linux", True), ("win32", False), ("darwin", False)])
def test_hardening_logs_nothing_on_success_or_off_linux(monkeypatch, caplog, platform: str, result: bool) -> None:
    from types import SimpleNamespace

    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "set_non_dumpable", lambda: result)
    monkeypatch.setattr(ui, "_hardened", None)
    monkeypatch.setattr(ui, "sys", SimpleNamespace(platform=platform))

    with caplog.at_level(logging.WARNING, logger=ui.logger.name):
        assert ui._harden_process() is result

    assert [record for record in caplog.records if record.name == ui.logger.name] == []


def test_the_page_imports_the_provider_sdks_once_per_run(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    fired: list[str] = []

    def recorder() -> None:
        fired.append("imported")

    monkeypatch.setattr(ui, "import_provider_sdks", recorder)
    monkeypatch.setattr(ui, "_sdks_imported", None)

    app = _app(tmp_path, monkeypatch).run()
    app.run()  # a rerun, as every page load is

    assert not app.exception
    assert fired == ["imported"]  # once per server process, not once per page load


def test_a_cache_clear_does_not_reset_the_run_slots(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch).run()
    slots = ui._slots
    st.cache_resource.clear()  # what a websocket clear_cache message does

    app.run()  # the next page load after that message

    assert not app.exception
    assert ui._slots is slots
    # The limits survived the clear: still at most one run per user.
    assert slots.admit("ana") is None
    assert "already have a run" in slots.admit("ana")
    slots.leave("ana")


def test_a_cache_clear_does_not_reset_the_run_quota(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch).run()
    quota = ui._quota
    st.cache_resource.clear()  # what a websocket clear_cache message does

    app.run()  # the next page load after that message

    assert not app.exception
    assert ui._quota is quota  # the same counter serves every page load


def test_a_cache_clear_does_not_start_a_second_cleanup_thread(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch).run()
    sweeper = ui._sweeper
    st.cache_resource.clear()  # what a websocket clear_cache message does

    app.run()  # the next page load after that message

    assert not app.exception
    assert ui._sweeper is sweeper  # the same thread serves every page load
    assert sum(thread.name == "doc-pipeline-cleanup" for thread in threading.enumerate()) == 1


def test_app_page_renders_rules_only_warning(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert any("No API key entered" in warning.value for warning in app.warning)
    assert app.button(key="start").disabled
    assert [box.value for box in app.sidebar.selectbox][:1] == ["none"]
    # A bare load reserves the session folder but does not create it.
    assert list(tmp_path.iterdir()) == []


def test_a_bare_anonymous_load_creates_no_folder_but_a_run_still_does(tmp_path: Path, monkeypatch) -> None:
    # Scripted visits that never upload, test models, or start a run leave
    # nothing for cleanup to sweep; a real run still gets its folder.
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert list(tmp_path.iterdir()) == []

    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert "result" in app.session_state
    assert [entry.name[:8] for entry in tmp_path.iterdir()] == ["session-"]


def test_a_page_load_refreshes_an_existing_session_folder_but_still_creates_none(tmp_path: Path, monkeypatch) -> None:
    # A page load touches the folder so cleanup does not sweep it while the
    # visitor waits for a run slot, yet a bare load still creates nothing.
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert list(tmp_path.iterdir()) == []  # the bare load created no folder

    workspace = Path(app.session_state["workspace"])
    ensure_workspace(workspace)
    stale = time.time() - 3600
    os.utime(workspace, (stale, stale))

    app.run()  # a rerun, as every page load is

    assert not app.exception
    mtime = workspace.stat().st_mtime
    assert mtime > stale  # the visit marked the folder as in use


def test_a_failed_workspace_creation_does_not_leak_the_run_slot(tmp_path: Path, monkeypatch) -> None:
    # The key is admitted into the run slots before the folder is made; if
    # making it fails, the slot is given back, so the next run is not refused.
    from doc_pipeline.app import session, ui

    app = _app(tmp_path, monkeypatch).run()
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()

    def failing(workspace: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ui, "ensure_workspace", failing)
    app.button(key="start").click().run()
    assert app.exception  # the failure surfaced instead of vanishing

    # The patch is replaced, not undone: monkeypatch.undo() would also drop
    # the fixture's other patches, including the workspaces env var.
    monkeypatch.setattr(ui, "ensure_workspace", session.ensure_workspace)
    app.button(key="start").click().run()

    assert not app.exception
    assert not any("already have a run" in warning.value for warning in app.warning)
    assert "result" in app.session_state  # the second run started


def test_page_splitting_choice_survives_adding_a_model(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()

    app.selectbox(key="provider_0").set_value("gemini").run()
    # The first model becomes the default for unclear pages.
    assert app.radio(key="splitting_models").value == 1
    app.radio(key="splitting_models").set_value("rules").run()
    app.selectbox(key="provider_1").set_value("openai").run()

    assert not app.exception
    assert app.radio(key="splitting_models").value == "rules"


def test_documents_read_at_once_appears_with_a_model(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()
    assert not [slider for slider in app.sidebar.slider if slider.key == "workers"]

    app.selectbox(key="provider_0").set_value("gemini").run()

    assert app.slider(key="workers").value == 3


def test_run_progress_shows_documents_time_left_and_retries() -> None:
    now = [0.0]
    progress = RunProgress(clock=lambda: now[0])
    progress.start_file(1, 2, "b.pdf")
    assert progress.snapshot() == (0.5, "Reading the pages of b.pdf (file 2 of 2) · 0:00 elapsed", "")

    progress.event(ProgressEvent("extracting", 0, 4))
    now[0] = 30.0
    progress.event(ProgressEvent("extracting", 1, 4))
    assert "left" not in progress.snapshot()[1]  # too early to tell
    now[0] = 40.0
    progress.event(ProgressEvent("extracting", 2, 4))
    progress.event(ProgressEvent("reply"))
    progress.event(ProgressEvent("retry", note="model_1: RateLimitError (HTTP 429); retrying in 20 s"))
    now[0] = 52.0
    fraction, text, note = progress.snapshot()

    assert fraction == pytest.approx((1 + 0.1 + 0.9 * 2 / 4) / 2)
    assert text == (
        "Extracting fields from b.pdf (file 2 of 2): 2 of 4 documents done · 0:52 elapsed · less than 2 min left · "
        "last model reply 12 s ago"
    )
    assert note == "model_1: RateLimitError (HTTP 429); retrying in 20 s"
    now[0] = 400.0
    progress.event(ProgressEvent("reply"))
    progress.event(ProgressEvent("extracting", 3, 4))
    assert progress.snapshot()[1:] == (
        "Extracting fields from b.pdf (file 2 of 2): 3 of 4 documents done · 6:40 elapsed · about 2 min left · "
        "last model reply 0 s ago",
        "",
    )
    progress.finish()
    assert progress.snapshot() == (1.0, "Writing the workbook…", "")


def test_a_running_job_is_picked_up_again_after_a_rerun(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import Future

    from doc_pipeline.app.ui import Job

    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    progress = RunProgress()
    result = run_uploads(config, keys, [Upload("batch.pdf", _pdf(["INVOICE\nInvoice No: A-1"]))], workspace, progress)
    assert progress.snapshot()[0] == 1.0
    # A job still working when the page reruns, as after a click mid-run.
    future: Future = Future()
    threading.Timer(0.3, future.set_result, [result]).start()
    app = _app(tmp_path, monkeypatch)
    app.session_state["job"] = Job(future, RunProgress(), rules_only=True)

    app.run()

    assert not app.exception
    assert "job" not in app.session_state
    assert "Results" in [header.value for header in app.subheader]
    assert app.session_state["result"] is result
    # Rules only: nothing was spent, so there is no cost box.
    assert not [metric for metric in app.metric if metric.label == "Estimated cost of this run"]


class _PageCrashesMidRun:
    """Progress stand-in: the page breaks on its second poll, so the job stays
    shown mid-run with its Cancel button, the way a live rerun would."""

    def __init__(self) -> None:
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return (0.0 if self.calls == 1 else 2.0, "Working…", "")  # 2.0 breaks st.progress


def test_the_cancel_button_stops_a_run_and_shows_what_finished(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import Future
    from dataclasses import replace

    from doc_pipeline.app.ui import Job

    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    result = replace(
        run_uploads(config, keys, [Upload("batch.pdf", _pdf(["INVOICE\nInvoice No: A-1"]))], workspace),
        stopped="the run was cancelled",
    )
    control = RunControl()
    future: Future = Future()

    def finish_when_cancelled() -> None:
        control.cancelled.wait(10)
        future.set_result(result)

    threading.Thread(target=finish_when_cancelled, daemon=True).start()
    app = _app(tmp_path, monkeypatch)
    app.session_state["job"] = Job(future, _PageCrashesMidRun(), rules_only=True, control=control)
    app.run()  # breaks mid-run, leaving the job and its Cancel button on the page

    app.button(key="cancel_run").click().run()

    assert control.cancelled.is_set()
    assert not app.exception
    assert "job" not in app.session_state
    assert app.session_state["result"] is result
    assert any("stopped early" in warning.value for warning in app.warning)


def test_a_run_cancelled_while_queued_says_so(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import Future

    from doc_pipeline.app.ui import Job

    future: Future = Future()
    future.set_result(None)  # RunSlots.run returns None for a queued run that was cancelled
    app = _app(tmp_path, monkeypatch)
    app.session_state["job"] = Job(future, RunProgress(), rules_only=True)

    app.run()

    assert not app.exception
    assert "The run was cancelled while waiting for a free slot." in [error.value for error in app.error]


def test_cost_box_shows_this_run_and_the_session_total(tmp_path: Path, monkeypatch) -> None:
    from concurrent.futures import Future
    from dataclasses import replace

    from doc_pipeline.app.ui import Job

    def usage(prompt: int, output: int) -> dict[str, int]:
        tokens = {"prompt_tokens": prompt, "output_tokens": output, "thoughts_tokens": 0, "cached_tokens": 0}
        return {**tokens, "total_tokens": prompt + output, "calls": 1}

    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, _choices(), workspace)
    result = run_uploads(config, keys, [Upload("batch.pdf", _pdf(["INVOICE\nInvoice No: A-1"]))], workspace)
    flash = ("gemini", "gemini-2.5-flash")
    # gemini-2.5-flash: 1M input at $0.30 plus 100k output at $2.50 = $0.55.
    result = replace(result, model_usage={flash: usage(1_000_000, 100_000), ("openai", "gpt-6-preview"): usage(900, 100)})
    future: Future = Future()
    future.set_result(result)
    app = _app(tmp_path, monkeypatch)
    app.session_state["job"] = Job(future, RunProgress(), rules_only=False)
    app.session_state["session_usage"] = {flash: usage(1_000_000, 0)}  # an earlier run: $0.30

    app.run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Estimated cost of this run"] == "$0.550+"
    assert metrics["This session so far"] == "$0.850+"
    assert any("gpt-6-preview: 1,000 tokens, price not set" in caption.value for caption in app.caption)
    assert app.session_state["session_usage"][flash]["calls"] == 2


def test_without_sign_in_set_up_the_app_refuses_to_run(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False).run()

    assert not app.exception
    assert any("Sign-in isn't set up" in error.value for error in app.error)
    assert not app.get("file_uploader")
    # Nothing is stored for a visitor who can't use the app.
    assert not list(tmp_path.iterdir())


def test_the_anonymous_switch_is_read_once_per_process(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert app.get("file_uploader")

    monkeypatch.delenv("DOC_PIPELINE_ALLOW_ANONYMOUS")
    app.run()  # a rerun, as every page load is

    assert not app.exception
    # Still anonymous: a later environment change (or a secrets edit Streamlit
    # copies into it) has no effect on this process.
    assert app.get("file_uploader")
    assert not app.error


def test_a_top_level_anonymous_secret_refuses_anonymous_mode(tmp_path: Path, monkeypatch) -> None:
    # _app sets the real variable too: that is what Streamlit does with a
    # top-level string secret, so the app cannot trust the value it sees.
    app = _app(tmp_path, monkeypatch)
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = "1"

    app.run()

    assert not app.exception
    assert any("DOC_PIPELINE_ALLOW_ANONYMOUS" in error.value for error in app.error)
    assert not app.get("file_uploader")
    # Refused before anything was stored for a visitor.
    assert not list(tmp_path.iterdir())


def test_an_anonymous_secret_keeps_refusing_after_it_is_removed(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch)
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = "1"
    app.run()
    assert any("DOC_PIPELINE_ALLOW_ANONYMOUS" in error.value for error in app.error)

    del app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"]
    app.run()

    assert not app.exception
    # Latched: anonymous use stays refused for the rest of the process.
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_an_integer_anonymous_secret_is_refused_too(tmp_path: Path, monkeypatch) -> None:
    # Any value type counts: a value that is not a string today can be edited
    # into one, which Streamlit would then copy into the environment.
    app = _app(tmp_path, monkeypatch)
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = 1

    app.run()

    assert not app.exception
    assert any("DOC_PIPELINE_ALLOW_ANONYMOUS" in error.value for error in app.error)
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_an_anonymous_secret_does_not_refuse_a_signed_in_account(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    # _app sets the real variable too: that is what Streamlit does with a
    # top-level string secret. The refusal is for anonymous use only, so with
    # sign-in set up a signed-in account is not sent to the refusal page.
    app = _app(tmp_path, monkeypatch)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = "1"
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))

    app.run()

    assert not app.exception
    assert not app.error
    assert app.button(key="sign_out").label == "Sign out"
    assert app.get("file_uploader")


def test_signed_out_visitors_only_see_sign_in(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Sign in with Google"]
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_allow_anonymous_lets_a_signed_out_visitor_use_the_app(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    # The allowlist stays in place for signed-in accounts; it cannot apply to
    # an anonymous visitor, and must not keep them out either.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allow_anonymous": True, "allowed_domains": ["example.com"]}
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")  # the proxy's appended entry

    app.run()

    assert not app.exception
    assert not app.error
    assert app.get("file_uploader")  # the app, anonymously
    assert "Sign in with Google" in [button.label for button in app.sidebar.button]  # the sign-in offer stays
    assert any("per IP address" in caption.value for caption in app.sidebar.caption)
    # The session folder is only made once a run starts, so a load makes none.
    assert list(tmp_path.iterdir()) == []


def test_allow_anonymous_with_one_unnamed_provider_still_offers_sign_in(tmp_path: Path, monkeypatch) -> None:
    # One provider set directly under [auth] comes back from sign_in_providers
    # as []: that is still sign-in set up, so the anonymous page keeps its
    # sidebar caption and its (nameless) sign-in offer.
    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = {
        "redirect_uri": AUTH["redirect_uri"],
        "cookie_secret": AUTH["cookie_secret"],
        "client_id": "id",
    }
    app.secrets["access"] = {"allow_anonymous": True}
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")

    app.run()

    assert not app.exception
    assert not app.error
    assert app.get("file_uploader")  # the app, anonymously
    assert "Sign in" in [button.label for button in app.sidebar.button]  # the nameless sign-in offer
    assert any("per IP address" in caption.value for caption in app.sidebar.caption)


def test_allow_anonymous_without_auth_opens_the_app(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["access"] = {"allow_anonymous": True}

    app.run()

    assert not app.exception
    assert not app.error  # not the fail-closed page
    assert app.get("file_uploader")
    assert app.button(key="start").label == "Start Processing"  # the page, not a refusal
    # No [auth] section, so there is no sign-in to offer either.
    assert not any("Sign in" in button.label for button in app.button)
    # A visit without a usable client IP is refused when it presses Start; a
    # bare load still creates no folder at all.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("access", [{}, {"allow_anonymous": False}, {"allow_anonymous": "true"}])
def test_allow_anonymous_off_or_not_a_boolean_keeps_the_sign_in_page(tmp_path: Path, monkeypatch, access) -> None:
    # Only a literal true opens the tier: anything else fails closed.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = access

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Sign in with Google"]
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_a_public_tier_visit_without_a_usable_ip_is_refused_on_start(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from doc_pipeline.app import audit

    # The secrets switch opens the tier, but no X-Forwarded-For arrives, and
    # the session fallback belongs to the local environment switch alone - so
    # there is nothing stable to count the visit against and it is refused.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["access"] = {"allow_anonymous": True}
    data = _pdf(["INVOICE\nInvoice No: A-1"])

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()
        app.button(key="start").click().run()  # shown again on the second press, logged once

    assert not app.exception
    assert any("without a usable one" in warning.value for warning in app.warning)
    assert "job" not in app.session_state and "result" not in app.session_state
    # The refusal comes before the workspace folder is made, so none exists.
    assert list(tmp_path.iterdir()) == []
    records = [
        record for record in caplog.records if record.name == audit.LOGGER_NAME and "reason=no_client_ip" in record.getMessage()
    ]
    assert [record.getMessage() for record in records] == ["event=run_refused user=anonymous reason=no_client_ip"]
    assert [record.levelno for record in records] == [logging.WARNING]


def test_a_public_tier_model_test_without_a_usable_ip_is_refused_too(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from doc_pipeline.app import audit

    # The model-test limits are keyed per IP as well, so the same refusal
    # applies before anything is sent to a provider.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["access"] = {"allow_anonymous": True}

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        app.selectbox(key="provider_0").set_value("gemini").run()
        app.text_input(key="api_key_0_gemini").set_value("key").run()
        [test_models] = [button for button in app.sidebar.button if button.label == "Test models"]
        test_models.click().run()

    assert not app.exception
    assert any("without a usable one" in warning.value for warning in app.warning)
    assert not app.success  # nothing was sent
    records = [record for record in caplog.records if record.name == audit.LOGGER_NAME]
    assert [record.getMessage() for record in records] == ["event=run_refused user=anonymous reason=no_client_ip"]
    assert [record.levelno for record in records] == [logging.WARNING]


def test_the_open_tier_without_sign_in_set_up_does_not_offer_it(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from doc_pipeline.app import audit

    # With the tier open but no [auth] section there is no sign-in to point
    # at, so none of the no-usable-IP texts may ask the visitor to sign in.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["access"] = {"allow_anonymous": True}
    data = _pdf(["INVOICE\nInvoice No: A-1"])

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        app.run()
        captions = [caption.value for caption in app.sidebar.caption]
        texts = [markdown.value for markdown in app.markdown]
        app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
        app.button(key="start").click().run()

    assert not app.exception
    # The refusal keeps the shared wording but names no sign-in anywhere.
    assert any("without a usable one" in warning.value for warning in app.warning)
    assert not any("sign in" in warning.value.lower() for warning in app.warning)
    assert any("no usable IP address" in caption for caption in captions)
    assert not any("needs sign-in" in caption for caption in captions)
    assert any("arrived without a usable IP address" in text for text in texts)
    assert not any("it needs to sign in to start runs" in text for text in texts)
    assert not [button for button in app.button if "Sign in" in button.label]
    records = [record for record in caplog.records if record.name == audit.LOGGER_NAME]
    assert [record.getMessage() for record in records] == ["event=run_refused user=anonymous reason=no_client_ip"]
    assert [record.levelno for record in records] == [logging.WARNING]


def test_the_open_tier_with_sign_in_set_up_keeps_its_sign_in_ways_out(tmp_path: Path, monkeypatch) -> None:
    # The same no-usable-IP visit on a host with an [auth] section keeps
    # today's wording: the refusal, the caption and the notice all point at
    # signing in, and the sign-in offer is there.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allow_anonymous": True}
    data = _pdf(["INVOICE\nInvoice No: A-1"])

    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert any("Please sign in to use the app" in warning.value for warning in app.warning)
    assert any("needs sign-in" in caption.value for caption in app.sidebar.caption)
    assert any(
        "it needs to sign in to start runs" in text for text in [markdown.value for markdown in app.markdown]
    )
    assert [button.label for button in app.button if "Sign in" in button.label] == ["Sign in with Google"]


def test_the_environment_switch_still_falls_back_to_the_session_key(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    # Locally (DOC_PIPELINE_ALLOW_ANONYMOUS=1) a visit without a proxy header
    # is keyed by its session, and starting a run still works and makes the
    # folder it needs.
    slots = _RecordingSlots()
    monkeypatch.setattr(ui, "_slots", slots)
    monkeypatch.setattr(ui, "client_ip", lambda headers: None)  # no proxy on this host
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch).run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert slots.admitted and slots.admitted[0].startswith("session:")
    assert "result" in app.session_state
    assert [entry.name[:8] for entry in tmp_path.iterdir()] == ["session-"]  # created by the run


def test_a_top_level_anonymous_secret_beats_the_secrets_switch(tmp_path: Path, monkeypatch) -> None:
    # The latch refuses anonymous use in every branch: with sign-in set up and
    # the secrets switch on, a top-level ANONYMOUS_ENV secret still means the
    # sign-in page, never anonymous admission.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allow_anonymous": True}
    app.secrets["DOC_PIPELINE_ALLOW_ANONYMOUS"] = "1"

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Sign in with Google"]
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_the_environment_switch_does_not_open_the_tier_when_sign_in_is_set_up(tmp_path: Path, monkeypatch) -> None:
    # DOC_PIPELINE_ALLOW_ANONYMOUS=1 stands for local development without an
    # [auth] section; with sign-in set up, a signed-out visitor signs in.
    app = _app(tmp_path, monkeypatch)  # sets the environment variable
    app.secrets["auth"] = AUTH

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Sign in with Google"]
    assert not app.get("file_uploader")
    assert not list(tmp_path.iterdir())


def test_the_notice_says_no_sign_in_is_needed_when_the_secrets_switch_is_on(
    tmp_path: Path, monkeypatch
) -> None:
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allow_anonymous": True}

    app.run()

    assert not app.exception
    assert any("No sign-in is needed" in text for text in [markdown.value for markdown in app.markdown])


def test_an_allowlist_with_the_open_tier_warns_once_per_process(tmp_path: Path, monkeypatch, caplog) -> None:
    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allow_anonymous": True, "allowed_domains": ["example.com"]}

    with caplog.at_level(logging.WARNING, logger=ui.logger.name):
        app.run()
        app.run()  # a second page load, as every visit is: still one warning

    assert not app.exception
    assert [record.getMessage() for record in caplog.records if record.name == ui.logger.name] == [
        ui.ALLOWLIST_ANONYMOUS_WARNING
    ]


def test_no_allowlist_warning_without_the_allowlist_or_without_the_switch(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    from doc_pipeline.app import ui

    # The switch is on, but there is nothing to warn about without a allowlist.
    open_only = _app(tmp_path, monkeypatch, anonymous=False)
    open_only.secrets["auth"] = AUTH
    open_only.secrets["access"] = {"allow_anonymous": True}
    open_only.run()
    assert not open_only.exception

    # The allowlist is set, but the tier stays closed, so the warning is beside the point.
    allowlist_only = _app(tmp_path, monkeypatch)
    allowlist_only.secrets["auth"] = AUTH
    allowlist_only.secrets["access"] = {"allowed_domains": ["example.com"]}
    allowlist_only.run()
    assert not allowlist_only.exception

    assert not [record for record in caplog.records if record.name == ui.logger.name]


def test_the_notice_names_the_anonymous_run_limits(tmp_path: Path, monkeypatch) -> None:
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    # No proxy header reaches the test page, so this visit counts per browser
    # session (the local environment switch's fallback) - and never per IP.
    assert any(
        "No sign-in is needed" in text
        and "10 per hour" in text
        and "40 per day" in text
        and "per browser session" in text
        and "per account" in text
        and "Start Processing" in text
        and "Test models" in text
        for text in [markdown.value for markdown in app.markdown]
    )
    texts = [markdown.value for markdown in app.markdown]
    assert not any("per IP address" in text for text in texts)
    assert any("per browser session" in caption.value for caption in app.sidebar.caption)
    assert not any("per IP address" in caption.value for caption in app.sidebar.caption)


def test_the_notice_says_per_ip_behind_a_proxy(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert any(
        "No sign-in is needed" in text and "per IP address" in text and "per account" in text
        for text in [markdown.value for markdown in app.markdown]
    )
    assert any("per IP address" in caption.value for caption in app.sidebar.caption)


def test_the_notice_does_not_say_no_sign_in_where_sign_in_is_required(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    # A default deployment (sign-in set up, the anonymous switch off) still
    # shows the run limits, but must not claim that no sign-in is needed.
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))

    app.run()

    assert not app.exception
    assert not app.error
    texts = [markdown.value for markdown in app.markdown]
    assert not any("No sign-in is needed" in text for text in texts)
    assert any(
        "Runs are limited to 10 per hour" in text and "40 per day" in text and "Start Processing" in text
        for text in texts
    )


def test_signed_in_accounts_are_checked_against_the_allowlist(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    def page(**claims: object):
        app = _app(tmp_path, monkeypatch, anonymous=False)
        app.secrets["auth"] = AUTH
        app.secrets["access"] = {"allowed_domains": ["example.com"]}
        monkeypatch.setattr(st, "user", _SignedIn(**claims))
        return app.run()

    allowed = page(email="boss@example.com", email_verified=True)
    assert not allowed.exception
    assert any("Signed in as boss@example.com" in caption.value for caption in allowed.sidebar.caption)
    assert allowed.button(key="sign_out").label == "Sign out"
    assert allowed.get("file_uploader")

    # Another domain, or an email the provider did not verify.
    for claims in ({"email": "stranger@other.example", "email_verified": True}, {"email": "boss@example.com"}):
        refused = page(**claims)
        assert not refused.exception
        assert any("is not on the list" in error.value for error in refused.error)
        assert not refused.get("file_uploader")


class _RecordingSlots(RunSlots):
    """RunSlots that remembers who was admitted, so a test can see the run key."""

    def __init__(self) -> None:
        super().__init__(running=2, waiting=4)
        self.admitted: list[str] = []
        self.left: list[str] = []

    def admit(self, user: str):
        self.admitted.append(user)
        return super().admit(user)

    def leave(self, user: str) -> None:
        self.left.append(user)
        super().leave(user)


def test_every_forwarded_line_is_read_before_the_last_entry_is_taken() -> None:
    from doc_pipeline.app import ui

    class Headers:
        """What StreamlitHeaders is: get_all sees repeated header lines."""

        def get_all(self, name: str) -> list[str]:
            return ["203.0.113.7, 198.51.100.9", "192.0.2.1"] if name == "X-Forwarded-For" else []

    # The appended second line carries the client address, so it wins.
    assert ui._client_ip(Headers()) == "192.0.2.1"
    # A plain mapping (as in the other tests) still works.
    assert ui._client_ip({"X-Forwarded-For": "203.0.113.7, 198.51.100.9"}) == "198.51.100.9"
    assert ui._client_ip({}) is None


def test_an_anonymous_run_is_keyed_by_the_client_ip(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    slots = _RecordingSlots()
    monkeypatch.setattr(ui, "_slots", slots)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert slots.admitted == ["ip:192.0.2.1"]
    assert "result" in app.session_state


def test_without_a_forwarded_header_the_run_is_keyed_by_the_session(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    app = _app(tmp_path, monkeypatch).run()  # creates the session folder the fallback names
    slots = _RecordingSlots()
    key = f"session:{app.session_state['workspace']}"
    assert slots.admit(key) is None  # its one run is in progress
    monkeypatch.setattr(ui, "_slots", slots)
    monkeypatch.setattr(ui, "client_ip", lambda headers: None)  # no proxy header on this host
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert slots.admitted == [key, key]  # the hold above and the refused run name the same key
    # And today's wording: the session key is one visitor, not a shared network.
    assert (
        "You already have a run in progress. Wait for it to finish, then start the next one."
        in [warning.value for warning in app.warning]
    )


def test_a_signed_in_run_keeps_the_account_key(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    slots = _RecordingSlots()
    monkeypatch.setattr(ui, "_slots", slots)
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert slots.admitted == ["https://accounts.google.com|42"]
    assert "result" in app.session_state


def test_an_ip_collision_says_the_network_has_a_run(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    slots = _RecordingSlots()
    assert slots.admit("ip:192.0.2.1") is None  # someone else on this network is running already
    monkeypatch.setattr(ui, "_slots", slots)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert (
        "Your network already has a run in progress (limits count per IP address). "
        "Sign in, or wait for it to finish."
    ) in [warning.value for warning in app.warning]
    assert slots.admitted == ["ip:192.0.2.1", "ip:192.0.2.1"]  # the same IP key held the slot and was refused


def test_a_quota_refusal_shows_the_wait_and_charges_no_slot(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    quota = RunQuota(1, 40, clock=time.monotonic)
    assert quota.admit("ip:192.0.2.1") is None  # this IP's hour is used up
    monkeypatch.setattr(ui, "_quota", quota)
    slots = _RecordingSlots()
    monkeypatch.setattr(ui, "_slots", slots)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.1")
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert any("try again in 60 minutes" in warning.value for warning in app.warning)
    assert slots.admitted == ["ip:192.0.2.1"]  # the slot place was held first...
    assert slots.left == ["ip:192.0.2.1"]  # ...and given back, so no limit was charged
    assert "job" not in app.session_state and "result" not in app.session_state
    assert not (Path(app.session_state["workspace"]) / "uploads").exists()  # no upload was read


def test_test_models_is_capped_by_its_own_quota_and_charges_only_it(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging

    from doc_pipeline.app import audit, ui

    monkeypatch.setattr(ui, "check_models", lambda config, keys: [("model_1", True, "gemini/test-model answered")])
    test_quota = ModelTestQuota(1, clock=time.monotonic)
    monkeypatch.setattr(ui, "_test_quota", test_quota)
    run_quota = RunQuota(10, 40, clock=time.monotonic)
    monkeypatch.setattr(ui, "_quota", run_quota)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.5")
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.selectbox(key="provider_0").set_value("gemini").run()
    app.text_input(key="api_key_0_gemini").set_value("key").run()
    [test_models] = [button for button in app.sidebar.button if button.label == "Test models"]

    test_models.click().run()  # the first test is admitted and charged to the test quota

    assert not app.exception
    assert any("answered" in success.value for success in app.success)
    assert len(test_quota._tests["ip:192.0.2.5"]) == 1
    assert run_quota._starts == {}  # the run quota was never touched by a test press

    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        test_models.click().run()  # the second press is refused at the cap
        test_models.click().run()  # shown again, logged once

    assert not app.exception
    assert any("already tested models 1 times" in warning.value for warning in app.warning)
    assert not app.success
    records = [record for record in caplog.records if record.name == audit.LOGGER_NAME]
    assert [record.getMessage() for record in records] == ["event=run_refused user=anonymous reason=test_quota"]
    assert [record.levelno for record in records] == [logging.WARNING]


def test_a_test_models_press_that_cannot_run_charges_no_quota(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    # "Other…" with an empty Model ID cannot build a config, so the press is
    # refused before the quota is charged; a press that does run a test is
    # charged exactly once.
    monkeypatch.setattr(ui, "check_models", lambda config, keys: [("model_1", True, "gemini/test-model answered")])
    test_quota = ModelTestQuota(1, clock=time.monotonic)
    monkeypatch.setattr(ui, "_test_quota", test_quota)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.5")
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.selectbox(key="provider_0").set_value("gemini").run()
    app.text_input(key="api_key_0_gemini").set_value("key").run()
    app.selectbox(key="model_0_gemini").set_value(ui.OTHER_MODEL).run()
    [test_models] = [button for button in app.sidebar.button if button.label == "Test models"]

    test_models.click().run()

    assert not app.exception
    assert any("Fix the settings first" in error.value for error in app.error)
    assert not app.success  # nothing was sent
    assert test_quota._tests == {}  # the unusable press charged nothing

    app.text_input(key="custom_model_0_gemini").set_value("gemini/test-model").run()
    test_models.click().run()

    assert not app.exception
    assert any("answered" in success.value for success in app.success)
    assert len(test_quota._tests["ip:192.0.2.5"]) == 1  # exactly one charge


def test_test_models_does_not_charge_the_run_quota(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "check_models", lambda config, keys: [("model_1", True, "gemini/test-model answered")])
    quota = RunQuota(2, 40, clock=time.monotonic)
    assert quota.admit("ip:192.0.2.5") is None  # one run of the hour is already spent
    monkeypatch.setattr(ui, "_quota", quota)
    monkeypatch.setattr(ui, "client_ip", lambda headers: "192.0.2.5")
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app = _app(tmp_path, monkeypatch)
    app.run()
    app.selectbox(key="provider_0").set_value("gemini").run()
    app.text_input(key="api_key_0_gemini").set_value("key").run()
    [test_models] = [button for button in app.sidebar.button if button.label == "Test models"]
    test_models.click().run()

    assert not app.exception
    assert any("answered" in success.value for success in app.success)
    # Back to rules only, so the run below makes no model request.
    app.selectbox(key="provider_0").set_value("none").run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert not any("try again in" in warning.value for warning in app.warning)
    assert "result" in app.session_state  # Start was only the hour's second run, so Test models never counted


def test_the_run_quota_limits_signed_in_account_keys_too(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from doc_pipeline.app import ui

    quota = RunQuota(1, 40, clock=time.monotonic)
    assert quota.admit("https://accounts.google.com|42") is None  # this account's hour is spent
    monkeypatch.setattr(ui, "_quota", quota)
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))
    data = _pdf(["INVOICE\nInvoice No: A-1"])
    app.run()
    app.file_uploader[0].set_value(("batch.pdf", data, "application/pdf")).run()
    app.button(key="start").click().run()

    assert not app.exception
    assert any("try again in 60 minutes" in warning.value for warning in app.warning)
    assert "result" not in app.session_state


def test_sign_out_cancels_the_job_before_deleting_its_workspace(tmp_path: Path, monkeypatch) -> None:
    import streamlit as st

    from concurrent.futures import Future

    from doc_pipeline.app import ui as ui_module
    from doc_pipeline.app.ui import Job

    control = RunControl()
    workspace = new_workspace(tmp_path)
    removals, logouts = [], []
    original = ui_module.remove_workspace

    def recording_remove(path: Path) -> None:
        removals.append(control.cancelled.is_set())  # was the job stopped first?
        original(path)

    monkeypatch.setattr(ui_module, "remove_workspace", recording_remove)
    monkeypatch.setattr(st, "logout", lambda: logouts.append(True))
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH
    app.secrets["access"] = {"allowed_domains": ["example.com"]}
    monkeypatch.setattr(st, "user", _SignedIn(email="boss@example.com", email_verified=True))
    app.session_state["workspace"] = str(workspace)
    app.session_state["job"] = Job(Future(), _PageCrashesMidRun(), rules_only=True, control=control)
    app.run()  # breaks mid-run, leaving the job and its Sign out button on the page

    app.button(key="sign_out").click().run()

    assert removals == [True]  # the job was cancelled before its folder went
    assert logouts == [True]
    assert not workspace.exists()
    assert not app.exception
    assert "job" not in app.session_state


def test_run_slots_allow_one_run_per_user_and_cap_the_server() -> None:
    slots = RunSlots(running=1, waiting=1)

    assert slots.admit("ana") is None
    assert "already have a run" in slots.admit("ana")
    assert slots.admit("ben") is None
    assert "server is busy" in slots.admit("cy")

    started, release, queued = threading.Event(), threading.Event(), threading.Event()

    def first_run() -> str:
        started.set()
        release.wait(5)
        return "ana done"

    with ThreadPoolExecutor(max_workers=2) as pool:
        ana = pool.submit(slots.run, "ana", first_run)
        assert started.wait(5)
        ben = pool.submit(slots.run, "ben", lambda: "ben done", queued.set)
        assert queued.wait(5)  # Ben waits for Ana's slot.
        release.set()
        assert (ana.result(5), ben.result(5)) == ("ana done", "ben done")

    # Places are given up when runs end, and when they fail.
    assert slots.admit("cy") is None

    def failing_run() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        slots.run("cy", failing_run)
    assert slots.admit("cy") is None


def test_the_run_quota_blocks_the_next_run_within_the_hour() -> None:
    now = [0.0]
    quota = RunQuota(2, 40, clock=lambda: now[0])

    assert quota.admit("ana") is None
    assert quota.admit("ana") is None
    refused = quota.admit("ana")
    assert refused == (
        "You have already started 2 runs in the last hour. Please try again in 60 minutes."
    )
    assert refused.code == "hour_quota"

    # The window is sliding: just inside it the cap still holds, at its end it frees.
    now[0] = 3600 - 1
    assert quota.admit("ana").code == "hour_quota"
    now[0] = 3600
    assert quota.admit("ana") is None


def test_the_run_quota_blocks_across_hours_at_the_day_cap() -> None:
    now = [0.0]
    quota = RunQuota(10, 3, clock=lambda: now[0])

    for start in (0, 3600, 7200):  # one run per hour: never near the hour cap
        now[0] = start
        assert quota.admit("ana") is None
    now[0] = 0
    assert quota.admit("ben") is None  # Ben's only start is at 0
    now[0] = 7200
    refused = quota.admit("ana")
    assert "try again in 1320 minutes" in refused  # until the first start leaves the day window
    assert refused.code == "day_quota"

    now[0] = 86400 - 1
    assert quota.admit("ana").code == "day_quota"
    now[0] = 86400
    assert quota.admit("ben") is None
    assert quota._starts["ben"] == [86400]  # a fully expired key is dropped, not kept forever
    assert quota.admit("ana") is None  # Ana's first start left the day window, so a place freed up


def test_the_run_quota_waits_are_reported_rounded_up_with_a_minimum_of_one_minute() -> None:
    now = [0.0]
    quota = RunQuota(1, 10, clock=lambda: now[0])
    assert quota.admit("ana") is None

    now[0] = 3600 - 61
    assert "try again in 2 minutes" in quota.admit("ana")  # 61 s rounds up to 2 minutes
    now[0] = 3600 - 1
    assert "try again in 1 minutes" in quota.admit("ana")  # never below one minute


def test_the_run_quota_counts_each_key_apart() -> None:
    now = [0.0]
    quota = RunQuota(1, 2, clock=lambda: now[0])  # per_day above per_hour: the hour cap fires first

    assert quota.admit("ana") is None
    assert quota.admit("ana").code == "hour_quota"
    assert quota.admit("ip:192.0.2.1") is None
    assert quota.admit("session:abc") is None
    now[0] = 86400
    assert quota.admit("ana") is None  # a new day for Ana says nothing about the others


def test_the_run_quota_purges_fully_expired_keys() -> None:
    now = [0.0]
    quota = RunQuota(1, 1, clock=lambda: now[0])

    assert quota.admit("ip:192.0.2.1") is None  # this key's day is now spent
    assert quota.admit("ip:192.0.2.1").code == "day_quota"
    now[0] = 60  # inside the purge interval: the spent key is kept for now
    assert quota.admit("ip:192.0.2.2") is None
    assert set(quota._starts) == {"ip:192.0.2.1", "ip:192.0.2.2"}

    now[0] = 86400
    assert quota.admit("ip:192.0.2.3") is None  # the purge runs inside admit...
    assert set(quota._starts) == {"ip:192.0.2.2", "ip:192.0.2.3"}  # ...dropping only the spent key
    assert quota.admit("ip:192.0.2.1") is None  # a dropped key starts over: nothing is remembered


def test_the_model_test_quota_caps_each_key_per_sliding_hour() -> None:
    now = [0.0]
    quota = ModelTestQuota(2, clock=lambda: now[0])

    assert quota.admit("ip:192.0.2.1") is None
    now[0] = 1800
    assert quota.admit("ip:192.0.2.1") is None
    refused = quota.admit("ip:192.0.2.1")
    assert "already tested models 2 times in the last hour" in refused
    assert "try again in 30 minutes" in refused
    assert refused.code == "test_quota"

    # The window is sliding: at the hour's end the first test frees a place.
    now[0] = 3600 - 1
    assert quota.admit("ip:192.0.2.1").code == "test_quota"
    now[0] = 3600
    assert quota.admit("ip:192.0.2.1") is None


def test_the_model_test_quota_counts_keys_apart() -> None:
    now = [0.0]
    quota = ModelTestQuota(1, clock=lambda: now[0])

    assert quota.admit("ip:192.0.2.1") is None
    assert quota.admit("ip:192.0.2.2") is None  # another network
    assert quota.admit("https://accounts.google.com|42") is None  # another account
    assert quota.admit("ip:192.0.2.1").code == "test_quota"


def test_the_model_test_quota_purges_fully_expired_keys() -> None:
    now = [0.0]
    quota = ModelTestQuota(1, clock=lambda: now[0])

    assert quota.admit("ip:192.0.2.1") is None  # this key's hour is now spent
    now[0] = 60  # inside the purge interval: nothing is dropped yet
    assert quota.admit("ip:192.0.2.2") is None
    assert set(quota._tests) == {"ip:192.0.2.1", "ip:192.0.2.2"}

    now[0] = 3600
    assert quota.admit("ip:192.0.2.3") is None  # the purge runs: key 1 left the hour window
    assert set(quota._tests) == {"ip:192.0.2.2", "ip:192.0.2.3"}  # key 2 is younger and stays
    now[0] = 4500  # 900 s after the last purge, the next admit purges again
    assert quota.admit("ip:192.0.2.4") is None
    assert set(quota._tests) == {"ip:192.0.2.3", "ip:192.0.2.4"}
    assert quota.admit("ip:192.0.2.1") is None  # a dropped key starts over: nothing is remembered


def test_a_queued_run_says_it_is_waiting() -> None:
    now = [0.0]
    progress = RunProgress(clock=lambda: now[0])
    progress.wait()
    now[0] = 65.0

    assert progress.snapshot() == (0.0, "Waiting for a free slot: other runs are in progress · 1:05 elapsed", "")
    progress.start_file(0, 1, "a.pdf")
    assert progress.snapshot()[1].startswith("Reading the pages of a.pdf")


def test_run_control_stops_on_cancel_or_time_limit_only_once_started() -> None:
    now = [0.0]
    control = RunControl(timeout_minutes=30, clock=lambda: now[0])
    control.check()  # not started and not cancelled: waiting for a slot is free

    control.start()  # the deadline begins when the run leaves the queue
    now[0] = 30 * 60 - 1
    control.check()
    now[0] = 30 * 60
    with pytest.raises(RunStopped, match="30-minute time limit"):
        control.check()

    cancelled = RunControl(clock=lambda: now[0])
    cancelled.check()
    cancelled.cancelled.set()
    with pytest.raises(RunStopped, match="the run was cancelled"):
        cancelled.check()


def test_a_queued_run_keeps_its_workspace_and_can_give_up(tmp_path: Path, monkeypatch) -> None:
    from doc_pipeline.app import session as session_module

    monkeypatch.setattr(session_module, "QUEUE_POLL_SECONDS", 0.01)
    slots = RunSlots(running=1, waiting=1)
    started, release = threading.Event(), threading.Event()
    workspace = new_workspace(tmp_path)
    hours_ago = time.time() - 7200
    os.utime(workspace, (hours_ago, hours_ago))
    keep_alives: list[float] = []

    def keep_alive() -> None:
        keep_alives.append(time.time())
        touch_workspace(workspace)

    def first_run() -> str:
        started.set()
        release.wait(5)
        return "first"

    with ThreadPoolExecutor(max_workers=3) as pool:
        pool.submit(slots.run, "ana", first_run)
        assert started.wait(5)
        ben = pool.submit(slots.run, "ben", lambda: "second", None, keep_alive)
        deadline = time.time() + 5
        while len(keep_alives) < 2 and time.time() < deadline:
            time.sleep(0.005)
        assert len(keep_alives) >= 2  # touched on every poll while queued
        # The queued run's folder was aged past the TTL, but the keep-alive saved it.
        assert cleanup_workspaces(60, root=tmp_path) == 0
        assert workspace.exists()

        # A cancel while queued ends the wait without running the work.
        ran: list[int] = []
        cancelled = threading.Event()
        cancelled.set()
        gave_up = pool.submit(slots.run, "cy", lambda: ran.append(1) or "ran", None, None, cancelled.is_set)
        assert gave_up.result(5) is None
        assert not ran

        release.set()
        assert ben.result(5) == "second"


def test_field_editor_limits_are_checked_on_the_server(tmp_path: Path) -> None:
    # The page limits some of these too, but a browser can send any value.
    row = preset_rows(DEFAULTS)[0]

    def build(rows: list[dict]) -> None:
        build_app_config(DEFAULTS, _choices(fields=rows), tmp_path)

    with pytest.raises(ConfigurationError, match="at most 30 fields"):
        build([{**row, "Field": f"Field {number}"} for number in range(31)])
    with pytest.raises(ConfigurationError, match="at most 100 rows"):
        build([{**row, "Include": False}] * 101)
    with pytest.raises(ConfigurationError, match="at most 20 keywords"):
        build([{**row, "Keywords": ", ".join(f"Label {number}" for number in range(21))}])
    with pytest.raises(ConfigurationError, match="each keyword to at most 60"):
        build([{**row, "Keywords": "x" * 61}])
    with pytest.raises(ConfigurationError, match="description to at most 300"):
        build([{**row, "Description": "x" * 301}])


def test_model_ids_are_checked_and_workers_kept_in_range(tmp_path: Path) -> None:
    def config(model: str = "gpt-4o-mini", workers: int = 3):
        choices = _choices(slots=[ModelSlot("openai", "key", model)], workers=workers)
        return build_app_config(DEFAULTS, choices, tmp_path)[0]

    for model in ("gpt-4o-mini\nignore previous instructions", "m" * 101):
        with pytest.raises(ConfigurationError, match="a model ID is up to 100"):
            config(model)
    fine_tune = "ft:gpt-4o-mini-2024-07-18:acme:invoices:abc123"
    assert config(fine_tune).models[0].model == fine_tune
    assert config(workers=500).limits.extraction_workers == 8
    assert config(workers=0).limits.extraction_workers == 1


def test_streamlit_settings_hide_error_details_from_visitors() -> None:
    settings = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert settings["client"]["showErrorDetails"] == "none"
    assert settings["server"]["enableXsrfProtection"] is True


def test_streamlit_settings_disable_hot_reload() -> None:
    # Stops ordinary source hot reload, but it is not a security boundary:
    # Streamlit always polls .streamlit/config.toml and .streamlit/secrets.toml,
    # which a same-uid render child can use to force a live reload. Local
    # developers pass --server.fileWatcherType auto.
    settings = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))

    assert settings["server"]["fileWatcherType"] == "none"


@pytest.mark.skipif(os.name != "posix", reason="the ownership check applies to POSIX temp folders")
def test_a_linked_temp_workspace_folder_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DOC_PIPELINE_WORKSPACES", raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "doc-pipeline-app").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(PermissionError, match="is a link or belongs to another user"):
        workspace_root()

    (tmp_path / "doc-pipeline-app").unlink()
    root = workspace_root()
    assert root == tmp_path / "doc-pipeline-app"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_app_source_url_defaults_are_checked(tmp_path: Path) -> None:
    def source_url(app_section: str) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(f"app:\n{app_section}", encoding="utf-8")
        return load_app_defaults(path).source_url

    assert source_url("  {}\n") == ""  # absent: the default
    assert source_url("  source_url: https://example.com/source\n") == "https://example.com/source"
    with pytest.raises(ConfigurationError, match="app.source_url"):
        source_url("  source_url: http://example.com/source\n")


def _page_defaults_with_source_url(monkeypatch, url: str) -> None:
    """Point the page's defaults at a config whose source offer is url (or none)."""
    from dataclasses import replace

    from doc_pipeline.app import ui

    monkeypatch.setattr(ui, "_defaults", {})
    monkeypatch.setattr(ui, "load_app_defaults", lambda _path: replace(DEFAULTS, source_url=url))


def test_the_main_page_offers_the_source_when_a_source_url_is_set(tmp_path: Path, monkeypatch) -> None:
    _page_defaults_with_source_url(monkeypatch, "https://example.com/source")

    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    assert "[Source code (AGPL-3.0)](https://example.com/source)" in [caption.value for caption in app.sidebar.caption]


def test_the_source_link_is_percent_encoded_so_markdown_cannot_break(tmp_path: Path, monkeypatch) -> None:
    _page_defaults_with_source_url(monkeypatch, "https://example.com/a(b)[c]<d>\\e`f\"g/h%20i")

    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    captions = [caption.value for caption in app.sidebar.caption if "Source code" in caption.value]
    assert len(captions) == 1
    caption = captions[0]
    assert caption.startswith("[Source code (AGPL-3.0)](https://example.com/") and caption.endswith(")")
    target = caption.removeprefix("[Source code (AGPL-3.0)](").removesuffix(")")
    for encoded in ("%28", "%29", "%5B", "%5D", "%3C", "%3E", "%5C", "%60", "%22"):
        assert encoded in target
    assert "%20i" in target and "%2520" not in target  # an existing escape is kept as it was
    assert not any(raw in target for raw in "()[]<>\\`\" ")


def test_the_main_page_names_the_licence_without_a_source_url(tmp_path: Path, monkeypatch) -> None:
    _page_defaults_with_source_url(monkeypatch, "")
    app = _app(tmp_path, monkeypatch).run()

    assert not app.exception
    captions = [caption.value for caption in app.sidebar.caption]
    assert "Licensed under the GNU AGPL-3.0" in captions
    assert not any("Source code" in caption for caption in captions)


def test_the_sign_in_page_offers_the_source(tmp_path: Path, monkeypatch) -> None:
    _page_defaults_with_source_url(monkeypatch, "https://example.com/source")
    app = _app(tmp_path, monkeypatch, anonymous=False)
    app.secrets["auth"] = AUTH

    app.run()

    assert not app.exception
    assert [button.label for button in app.button] == ["Sign in with Google"]
    assert "[Source code (AGPL-3.0)](https://example.com/source)" in [caption.value for caption in app.sidebar.caption]


def test_the_no_auth_refusal_page_offers_the_source(tmp_path: Path, monkeypatch) -> None:
    _page_defaults_with_source_url(monkeypatch, "https://example.com/source")

    app = _app(tmp_path, monkeypatch, anonymous=False).run()

    assert not app.exception
    assert any("Sign-in isn't set up" in error.value for error in app.error)
    assert "[Source code (AGPL-3.0)](https://example.com/source)" in [caption.value for caption in app.sidebar.caption]
