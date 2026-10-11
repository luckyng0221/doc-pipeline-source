# PDF Splitter & Extractor

Upload PDF batches, choose the fields you want, and download one Excel workbook with a row per document. Each batch is split into separate documents (invoices, receipts, statements, and so on), the chosen fields are extracted, and duplicates and failures are flagged.

AI models are optional and are called only when local checks cannot settle a question:

1. **Splitting:** local rules read the PDF text layer first. "Page 1 of 3" style markers, a finished page count, or a change of title line (for example INVOICE to OFFICIAL RECEIPT) decide most pages, and blank pages are skipped. Only pages the rules cannot decide go to a model, packed into as few requests as possible. Without a model, those pages are guessed and marked as guesses.
2. **Extraction:** with models, the first model reads each document. When every value it returns is printed in the PDF's own text, that answer is kept and no other model is called. Otherwise the second model is called, and the third (arbiter) only when the first two disagree. Without a model, values are read from the text next to each field's labels (keywords).

It runs as a web page (`streamlit_app.py`) where people sign in, upload PDFs, choose fields, optionally enter up to three of their own API keys, and press **Start Processing**.

## Project layout

| Path | What it holds |
|---|---|
| `streamlit_app.py` | Entry point for `streamlit run` and Streamlit Community Cloud |
| `src/doc_pipeline/` | The pipeline; `app/` holds the upload page and its helpers |
| `tests/` | Offline tests (`python3 dev.py pytest`) |
| `docs/` | Internal reports: the code review (`REVIEW.md`) and the security review (`SECURITY_REVIEW.md`) |
| `samples/` | Your own test PDFs (git-ignored) |
| `config.yaml` | App settings: limits, model suggestions, prices, field presets |
| `.streamlit/config.toml` | Streamlit server settings |
| `pyproject.toml` | Package metadata and dependency ranges |
| `requirements.txt` | The hash-pinned lock that Community Cloud installs |
| `requirements-dev.txt` | The hash-pinned lock that `dev.py` installs (app and dev extras) |
| `.githooks/` | The git hooks (pre-commit, pre-merge-commit, commit-msg, pre-push): block documents and scan for secrets |
| `dev.py` | Runs commands in the project's venv |

The files at the top level stay there because Streamlit, Community Cloud, pip, and pytest look for them there.

## Install

Install Python 3.11 or newer. Then run every command through `dev.py` from the project folder:

```bash
python3 dev.py pytest
```

On first use, `dev.py` creates a `.venv` in the project folder (git-ignored). It installs the exact, hash-checked versions from `requirements-dev.txt`, then the project itself (editable, with no extra downloads), and tries to turn on the git hooks. That fails inside Claude Code's sandbox, which can't write `.git/config`; run `git config core.hooksPath .githooks` yourself there. A venv only works on the machine that made it, so each clone builds its own. When the venv is missing, broken, or out of date with `pyproject.toml`, `dev.py` rebuilds or reinstalls it. `python3 dev.py` with no arguments prints the venv's Python path, for editors that need it. VS Code's Python extension finds `.venv` by itself. Set `$WORKON_HOME` to keep the venv in `$WORKON_HOME/doc-pipeline` instead. The install is editable, so changes to the code take effect without reinstalling.

The `app` extra pins pandas below 3.0: Windows Smart App Control blocks the DLLs in the brand-new pandas 3 wheels, while pandas 2.x loads normally. It also needs Streamlit 1.65 or newer, and the core install needs google-genai 1.75 or newer for Gemini 3 thinking levels. `dev.py` reinstalls whenever `pyproject.toml` or `requirements-dev.txt` changes, so a venv with older versions is upgraded automatically.

## Working on more than one PC

The code lives in a private GitHub repository. On a new PC (WSL2), clone it into the Linux filesystem, not under `/mnt/c`:

```bash
sudo apt install -y git gh python3-venv   # dev.py needs Python 3.11+ with venv support
gh auth login
mkdir -p ~/projects && cd ~/projects
gh repo clone <owner>/doc-pipeline
cd doc-pipeline
git config core.hooksPath .githooks   # turns on the git hooks
python3 dev.py pytest
```

`git pull` before you start, and commit and `git push` when you are done. `samples/` and secrets are git-ignored, so copy them separately.

## Upload app

```bash
DOC_PIPELINE_ALLOW_ANONYMOUS=1 python3 dev.py streamlit run streamlit_app.py
```

The page opens at http://localhost:8501. To try it, put a PDF in `samples/` (git-ignored, so it is never published) and upload it.

The app requires sign-in and refuses to run until sign-in is set up (see [Deploying](#deploying-to-streamlit-community-cloud)) — unless the owner opens the anonymous tier with `[access] allow_anonymous` in the secrets (also under Deploying). `DOC_PIPELINE_ALLOW_ANONYMOUS=1` skips sign-in for local use; never set it on a public server. To test sign-in locally instead, put the `[auth]` secrets in `.streamlit/secrets.toml` (git-ignored) with `redirect_uri = "http://localhost:8501/oauth2callback"`.

1. **AI models (sidebar, optional).** Up to three slots, each with a provider (Gemini, OpenAI, or GLM), the user's own API key, a model (a suggestion or any model ID), and a reasoning effort:
   - **Model 1** reads every document.
   - **Model 2** is asked only when Model 1's answer can't be confirmed from the PDF text (or on every document with *Second opinion: Always*).
   - **Model 3** settles disagreements between the first two.
   - **Page splitting** chooses whether a model decides the pages the rules can't, or whether those are guessed. Once a model is set, Model 1 is the default, and the choice is kept when more models are added or removed.
   - **Documents read at once** (1–8, default `limits.extraction_workers`) sets how many documents are sent to the models at the same time. Higher is faster. Lower it on free API tiers with low rate limits. It never changes the results.
   - **Test models** sends one tiny request per model to check the key, the model ID, and whether the model accepts the chosen effort.
2. **Fields to extract.** An editable table seeded with common fields. Users tick fields, change their type, or add their own. *Keywords* are the labels printed next to a value (for example "Invoice No"); they drive extraction when no key is entered and are given to the models as hints. *Duplicate key* fields form the duplicate fingerprint.
3. **Upload PDFs**, then press **Start Processing**. Nothing runs until then. While the run works, the progress bar shows:
   - the step and how many documents are done
   - the elapsed time and, after the first two documents, an estimate of the time left
   - how long ago a model last replied
   - a note when a model is rate-limited and the app is waiting to retry

   The run happens in the background, so changing a setting or clicking elsewhere on the page does not stop it.
4. **Results.** The page shows counts, a preview table, and any failed files with the reason. Users can download the Excel workbook (all files in one) and a zip of the split PDFs, with one folder per uploaded file (repeated file names get `_2`, `_3`). **Start over** deletes the session's files.

   With API keys, a cost box shows the **estimated cost of this run** and the **total for this session**. The session total adds up runs until the page is reloaded, and **Start over** keeps it. *Cost by model* lists requests, tokens, and cost per model.
   - **What counts:** every request the run actually made, including page-splitting requests and requests for files that then failed. A file whose results were reused (same file, same settings) costs nothing.
   - **Prices:** list prices from `app.prices` in `config.yaml`. Thinking tokens count as output. Free tiers and cached-token or batch discounts are not taken into account, so the figure can be a little high; the provider's billing page has the exact amount.
   - **Models without a price:** a model with no entry in `app.prices`, such as an ID typed in with *Other…*, is listed with its tokens and "price not set". The total then leaves it out and ends in `+`.

**Without API keys** the app still runs: documents are split by the boundary rules and fields are filled by label matching. The page warns clearly that these results may be less accurate. Label matching depends on the scan's text layer and layout, and many scanners write labels and values on separate lines. It cannot read image-only scans at all.

Reasoning effort per provider:

| Provider | Choices | Sent as |
|---|---|---|
| Gemini | default, off, minimal, low, medium, high | `thinking_level` (Gemini 3), `thinking_budget` (Gemini 2.x; off = 0) |
| OpenAI | default, none, minimal, low, medium, high | `reasoning_effort` (reasoning models only; leave default for gpt-4o-mini). o-series and GPT-5 models are never sent a `temperature`, since they accept only their default. |
| GLM | default, off, on | `thinking: enabled/disabled` |

### Privacy and public use

The app is built to be shared (for example as a public beta):

- **Sign-in:** people sign in with Google or Microsoft through Streamlit's built-in sign-in (`st.login`). An optional allowlist limits the app to listed emails or domains. Without sign-in set up, the app refuses to run. **Sign out** forgets the session's API keys and deletes its files.
- **Keys:** the keys people type are sent to the server and kept in its memory for their session only. They are never written to disk or logs, and the app never reads keys from the server's environment. The page tells users to use a dedicated key with a spending limit.
- **Isolated sessions:** each session gets its own temporary folder, including its own duplicate registry. Duplicate checks cover only that user's uploads. Folders are deleted on **Start over** or **Sign out**, or after `app.workspace_ttl_minutes` — a background thread also sweeps expired folders every 15 minutes, so cleanup does not depend on someone loading a page. They live in the system temp folder unless the `DOC_PIPELINE_WORKSPACES` environment variable points elsewhere.
- **Limits:**
  - `app:` in `config.yaml` sets files per run, MB per file, pages per PDF, and how many runs the server works on at once (`max_concurrent_runs`) or queues (`max_waiting_runs`). Each user has one run at a time.
  - A run also stops at `app.run_timeout_minutes` (60 by default): the files it had finished are kept and the rest are reported as not run. A **Cancel run** button stops a run the same way: no new model request starts, and one already in flight finishes first (up to the provider timeout).
  - Each visitor may start at most `app.max_runs_per_hour` (10 by default) runs per sliding hour and `app.max_runs_per_day` (40 by default) per day. When signed in, the runs are counted per account; without sign-in, per IP address — the last `X-Forwarded-For` entry, validated with Python's `ipaddress` module, with IPv6 counted per /64 network — so people behind one office or carrier network share one count. Locally (`DOC_PIPELINE_ALLOW_ANONYMOUS=1`), a visit whose header names no usable address is counted per browser session instead; on the public anonymous tier it is refused (logged as `run_refused reason=no_client_ip` at WARNING) rather than keyed by something a reload can change. **Test models** has its own hourly cap, `app.max_model_tests_per_hour` (20 by default), counted the same way and never taken from the run limits. The counters are in memory, so a server restart resets them.
  - `.streamlit/config.toml` caps the upload size and hides error details from visitors.
  - Untrusted PDFs are bounded too: large pages render at a lower resolution, images over 100 megapixels are refused, page text is cut at 20,000 characters, and the field table and model IDs are checked on the server. Each PDF is also rendered in its own disposable child process (POSIX only; other platforms render in-process), killed at `limits.render_timeout_seconds` (120 seconds by default) and capped at `limits.render_child_memory_mb` (1024 MB), so a slow or hostile PDF fails on its own instead of stalling or crashing the server for everyone. Only rendering is isolated that way: text extraction and splitting still open the PDF in the server process (a parser bug hit there runs inside the server, with every session's keys) and reopen the uploaded file after the child has run, so a compromised child could change what they parse, and the child runs as the same user under those resource limits — bounded, not sandboxed (there is no OS sandbox such as seccomp or Landlock). A compromised child could still write anything the server can write — this app's own code, the provider packages in its virtual environment, another visitor's temporary folder — and read the server's `.streamlit/secrets.toml` (the sign-in and cookie secrets) straight away. Its writes to the server's code or config can take effect without a restart: whatever the config says, Streamlit always polls `.streamlit/config.toml` and `.streamlit/secrets.toml`, so the child can touch either file to force a live reload of the server's code and config. Turning the file watcher off and importing the provider SDKs' top-level modules at start-up only narrow this; they are not a security boundary. Running the render child under a dedicated user account of its own, or a Landlock policy, would close this gap; it is not done.
- **What leaves the server:** with keys, pages are sent to the chosen AI provider (Google, OpenAI, or Z.ai / Zhipu; the BigModel endpoint is in China) and billed to the user's own account. On Gemini's free tier, Google may use submitted content to improve its products. Without keys, nothing leaves the server. The OpenAI and Gemini endpoints are fixed in the app's code and the GLM endpoints come from `config.yaml`, so the server's environment cannot redirect requests to another service or set their proxy or certificate authority. The provider SDKs and Python's TLS library still read a few variables, for example (verified on the installed versions, not an exhaustive list) `OPENAI_CUSTOM_HEADERS`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_LOG`, `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_GENAI_USE_ENTERPRISE`, `GOOGLE_GENAI_CLIENT_MODE`, `SSLKEYLOGFILE`, `SSL_CERT_FILE` and `SSL_CERT_DIR`. Some are serious: `OPENAI_LOG=debug` makes the OpenAI SDK log requests (prompts, page text and images), `GOOGLE_GENAI_CLIENT_MODE=record` writes Gemini requests and responses to disk, and `OPENAI_CUSTOM_HEADERS` can replace the `Authorization` header. Only someone who can write the server's secrets can plant them (see the isolation limits above).
- **Hidden text:** a PDF from someone else can carry text meant to mislead the model. Values are confirmed against the page's text with any detected hidden text left out, and detection is best-effort: hidden text (invisible, near-transparent, or under about 2 pt at the rendered size) is detected and such documents are never marked `TEXT_VERIFIED` — a second model is asked when one is configured (with a single model the result is `SINGLE_MODEL`) and a "Hidden text found on page N" remark is added — while an OCR layer over a scanned image counts as visible; detection has limits (clip-only text, for one, is not detected and stays in the text values are confirmed against; see `docs/SECURITY_REVIEW.md`), so the page still tells users to confirm bank details with the vendor before paying.
- **Audit lines:** the server keeps content-free audit lines of run starts and ends and refusals, with a pseudonymous id per visitor — never an email, a file name, or document content (see **Logs** under *Configuration notes*).

### Deploying to Streamlit Community Cloud

1. Put the project in a **private** GitHub repository. `.gitignore` already excludes secrets, key files, `samples/`, and document, image, and common archive formats; check that none of your documents are committed. In the repository's settings, turn on Dependabot alerts. Secret scanning, push protection and private vulnerability reporting aren't available for a private repository on a personal account, so a local check is all there is: the hooks in `.githooks/` are a local check. **pre-commit** (and **pre-merge-commit**, for clean merges) checks what is staged: documents (by name, by `.gitignore`, and by file signature), files detect-secrets would skip (UTF-16, other non-UTF-8 files, Git LFS and filtered files), the detect-secrets settings recorded in `.secrets.baseline`, this project's own OpenAI, Gemini, GLM, Google OAuth, AWS, Microsoft and private-key patterns, and detect-secrets itself. **commit-msg** checks the message for the same key patterns. **pre-push** re-checks the full content and message of every commit that is not yet on a remote-tracking ref, plus a detect-secrets scan, so a commit made with the hooks off is still caught before it leaves the machine. pre-push also scans annotated tag messages, refuses tags that point at trees or blobs, and checks a commit that has no baseline of its own against the pushed tip's baseline, with a warning. Still out of reach: `--no-verify` on a commit or push, a clone where the hooks aren't turned on, commits already on a remote-tracking ref (they are not re-scanned), and anything detect-secrets' detectors and these patterns don't know. It isn't a full replacement for GitHub's secret scanning. A stale baseline (line numbers moved, a recorded secret removed, or a new detect-secrets version) is allowed with a warning (the hook then prints the command to refresh it). For a detect-secrets false positive, add `# pragma: allowlist secret` to that line, or update the baseline with `python3 dev.py detect_secrets scan --no-verify --baseline .secrets.baseline` (`--no-verify` keeps it from sending suspected secrets to their providers). The pragma only silences detect-secrets: a hit from the project's own key patterns can't be allowlisted, so remove the key or build test values at runtime (for example by concatenation). A baseline update accepts every current finding, so check `git diff .secrets.baseline` before committing it. A baseline that drops a detector, adds a filter or changes the version is blocked until you have reviewed it. Accepting a changed baseline setting (for example after a detect-secrets upgrade, which changes the version or the detector list) means editing `PINNED_PLUGINS`, `PINNED_FILTERS` or `BASELINE_VERSION` in `.githooks/check_staged.py`. That is the user's decision, not an agent's. The pre-push check needs git 2.40 or newer (`check-attr --source`). Keep the internal reports in `docs/` out of any public repository.
2. In Streamlit Community Cloud, create an app from the repository with `streamlit_app.py` as the main file. `requirements.txt` is a lock: exact versions with hashes. The app itself loads from `src/`.
3. Set up Google sign-in (about 15 minutes):
   1. In Google Cloud Console, open APIs & Services → Credentials → Create credentials → OAuth client ID → Web application.
   2. Add the authorized redirect URI `https://<your-app>.streamlit.app/oauth2callback`. For local tests, also add `http://localhost:8501/oauth2callback`.
   3. Put the client ID and secret in the app's **Secrets** settings, not in the repository. For `cookie_secret`, paste the output of `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`:

      ```toml
      [auth]
      redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
      cookie_secret = "<long random string>"

      [auth.google]
      client_id = "<client ID>"
      client_secret = "<client secret>"
      server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

      # Optional: only these people may use the app. Leave out for any signed-in account.
      [access]
      allowed_emails = ["you@example.com"]
      allowed_domains = ["yourcompany.com"]
      ```

   Microsoft sign-in works the same way under `[auth.microsoft]`, with a tenant-specific `server_metadata_url` (`https://login.microsoftonline.com/<tenant>/v2.0/.well-known/openid-configuration`). The allowlist only accepts emails the provider marks as verified, and Microsoft usually does not, so use Google when you set an allowlist.

   To also let people use the app without an account, add `allow_anonymous = true` to the `[access]` section, as its own section (a line appended below another `[section]` belongs to that section) and with `true` unquoted:

   ```toml
   [access]
   allow_anonymous = true
   ```

   Visitors who are not signed in can then upload and run too, with their runs counted per IP address. This works with or without an `[auth]` section: without one, nobody signs in. Where sign-in is set up, it stays available and unchanged, and the allowlist still guards signed-in accounts only — an anonymous visitor has no email to check it against, so it is not applied to them. Three things to know before turning it on:

   - **Per-IP counting is only sound behind a proxy that appends the client address to `X-Forwarded-For`.** Community Cloud is expected to do this, but verify it before turning `allow_anonymous` on. The app trusts the last entry, so behind such a proxy a client's own header cannot change its key. With the app exposed directly — no appending proxy in front — a client can send whatever header it likes and pick a fresh `ip:` key per request, so do not open the anonymous tier without one.
   - **Check once after deploying** that the proxy's header really reaches the app (`st.context.headers`) and carries the client address last: load the app signed out and try it. A missing or unusable header shows up in the logs as `run_refused reason=no_client_ip` at WARNING (and the visitor is asked to sign in) — but a *constant* last entry does not: if two different clients show the same last `X-Forwarded-For` entry, a second proxy sits in front of the one that appends, and every visitor shares one key. Then confirm a client cannot forge its own key: set `app.max_model_tests_per_hour` to 1, use **Test models** once, and from a new tab send a request with a forged `X-Forwarded-For` header — a plain browser tab cannot set request headers, so use a browser extension or tool that can (for example a header-modifying extension). The forged address must still be refused (`run_refused reason=test_quota`); if it gets a fresh test instead, the last entry is client-controlled and `allow_anonymous` must stay off.
   - **The `[access]` allowlist does not restrict anonymous visitors**, so the app is open to anyone while `allow_anonymous` is on, allowlist or not. When both are configured, the app logs a one-time WARNING per process saying so.

   This secrets switch is the only way to open the public tier: never set `DOC_PIPELINE_ALLOW_ANONYMOUS` on a server (see [Upload app](#upload-app)); that variable is for local development.
4. Use MFA on your GitHub, Streamlit, Google, and Microsoft accounts. If a secret ever leaks, rotate it first.

Review `app.models` in `config.yaml` so the suggested model IDs match what the providers currently offer. Users can always type another model ID.

### Dependencies and security updates

`requirements.txt` pins every package, with hashes, so a rebuild never picks up a new (possibly compromised) release by itself. After changing `pyproject.toml`, or every few weeks to pick up fixes, regenerate both locks and check them against the known-vulnerability database (`requirements-dev.txt` is the one `dev.py` installs and is constrained to the versions in `requirements.txt`). `--python-version 3.11` keeps the locks valid for Python 3.11+ whichever Python runs uv, and `--no-python-downloads` stops uv from downloading CPython 3.11 when it isn't installed:

```bash
python3 dev.py uv pip compile pyproject.toml --extra app --universal --python-version 3.11 --no-python-downloads --generate-hashes --upgrade -o requirements.txt
python3 dev.py uv pip compile pyproject.toml --extra app --extra dev --universal --python-version 3.11 --no-python-downloads --generate-hashes --upgrade -c requirements.txt -o requirements-dev.txt
python3 dev.py pip_audit -r requirements.txt --disable-pip
python3 dev.py pip_audit -r requirements-dev.txt --disable-pip
python3 dev.py pytest
```

The `--upgrade` on the dev lock refreshes the dev tools too. `dev.py` warns (and still runs your command) when the installed packages don't satisfy `pyproject.toml`. `tests/test_locks.py` checks that the two locks agree with each other and with `pyproject.toml`.

## Results, duplicates, and errors

The workbook contains these sheets:

- **Documents:** one row per detected document. It includes how many models were called and a **Remarks** column explaining where the document starts and why each extra model was called. Conflicts are shaded red; rows to review are yellow.
- **Pages:** how each page was classified, and whether a rule, a model, or a guess decided it.
- **Model audit:** each model's structured answer.
- **Token usage:** tokens per model call.
- **Run info:** the mode (which models or rules only) and page and document counts per source, plus token totals.
- **Failed files:** each file that could not be processed, with the reason (only when there are failures).

Validation statuses:

- `TEXT_VERIFIED`: the first model's values were all found in the PDF's text (with any detected hidden text left out), so no other model was called; a document with hidden (non-OCR) text never gets this status — a second model is asked when one is configured (with a single model the result is `SINGLE_MODEL`) and a "Hidden text found" remark is added.
- `SINGLE_MODEL`: only one model answered (one configured, or the others failed) and its values could not all be confirmed from the text. Check these.
- `RULES_ONLY`: no AI model; values come from label matching. Check these.
- `CONSENSUS_2_PRIMARIES`: the first two models agree on every populated field.
- `CONSENSUS_3`: the arbiter was called and all models agree.
- `CONSENSUS_2_OF_3`: a majority agrees on each populated field, with a dissent or abstention.
- `INCOMPLETE_VALIDATION`: only two models returned usable results and they agree.
- `CONFLICT`: at least one populated field has no two-model majority.
- `FAILED`: no usable model answer (for example, the requests kept failing). The values are empty, and the Remarks column says why. Uploading the file again retries it.

A value counts as confirmed only when it is printed in the document's text:

- Amounts must match exactly, allowing common OCR slips such as `I` for `1`.
- Reference numbers must match whole, apart from OCR slips, spacing, and punctuation. `INV-2024-001` is not confirmed by `INV-2024-0015`.
- Names may differ by one OCR letter per word.
- Zero amounts never count.
- A numeric date such as 02/12/2024 counts only when its order is unambiguous or the date order is set. A month and year alone ("March 2024") never confirm a day.

This proves a value is on the page, not that the model picked the right one (for example a subtotal instead of the total). Set *Second opinion* to *Always* in the app to have two models vote on every document.

**Duplicates:**
- An identical file processed again (same SHA-256 and settings) reuses the earlier result without new model calls and is flagged `DUPLICATE`.
- Documents whose duplicate-key values all match, or whose pages look alike, are flagged `POSSIBLE_DUPLICATE` and still processed. With no *Duplicate key* field ticked, only look-alike pages are flagged.

**Errors:** a PDF is listed under Failed files, with the reason, in these cases:
- It is unreadable, password-protected, too large, or has too many pages.
- The boundary model fails or is unsure.
- None of its documents gets a usable answer.

When only some documents get no usable answer, the PDF still produces results and only those rows are marked `FAILED`. With an arbiter configured, fewer than two usable answers counts as no usable answer. After 3 documents in a row fail (for example, an exhausted quota), the remaining documents are not sent and are marked `FAILED` too. A PDF with `FAILED` rows is not cached, so uploading it again reads it again.

Other files in the same run carry on. Excel text is sanitized against formula injection, including user-defined field names. Workbooks are written atomically.

## Configuration notes

- **Cache key:** within a session, cached results are reused only when every setting that can change results is unchanged. Retry, size, duplicate-matching, and parallel-request (`extraction_workers`) limits are not part of that check.
- **Prices (`app.prices`):** USD per 1M input and output tokens, by model ID, for the cost estimate. Each suggested model in `app.models` needs a price (a test checks this). Prices change, so check them against the providers' pricing pages now and then. Gemini 3.8 Flash's introductory rate ends on 2026-12-31.
- **Gemini input:** Gemini models can read native PDF pages (`input: pdf`, a flat ~258 tokens per page instead of ~1,500–2,000 for a rendered image). The app uses this automatically for Gemini.
- **`boundary_rules`:**
  - `blank_pages: ignore` skips blank pages; `separator` also starts a new document after one. Use `separator` only when blank sheets are inserted between documents, not for duplex scans.
  - `fallback` decides unclear pages when there is no boundary model.
  - `titles` replaces the built-in title lines.
- **Logs:** the app writes no OCR text, prompts, page images, or keys to logs (a planted `OPENAI_LOG` variable would make the OpenAI SDK log requests; see *Privacy and public use*). The downloaded workbook's audit sheet keeps structured model output, so protect it like the source PDFs. The server also keeps content-free audit lines (see *Privacy and public use*). On Community Cloud you read them in your app's **Manage app → logs** view; when running the server yourself they go to its stderr. Each event is one line, for example:

  ```
  2026-10-11 09:30:12,123 INFO    doc_pipeline.audit: event=run_started user=8f14e45fceab files=3 bytes=1048576 mode=rules
  2026-10-11 09:34:55,456 INFO    doc_pipeline.audit: event=run_ended user=8f14e45fceab outcome=finished seconds=283 files_done=3 files_failed=0 documents=7
  2026-10-11 09:31:02,789 WARNING doc_pipeline.audit: event=run_refused user=unknown reason=too_many_files
  ```

  `user=` is a pseudonym, never an email: the first 12 hex characters of HMAC-SHA256 of the person's lowercased email, keyed with your `[auth] cookie_secret` (`anonymous` for any visitor who is not signed in — including on the public anonymous tier, not only where sign-in is off — and `unknown` when there is no key). To see whether a line belongs to someone, recompute the id for their address — the script asks for the email and reads the secret with `getpass`, so neither lands on the command line or in the shell history:

  ```bash
  python3 -c "import getpass, hmac, hashlib; email = input('email: '); secret = getpass.getpass('cookie_secret: '); print(hmac.new(secret.encode(), email.strip().lower().encode(), hashlib.sha256).hexdigest()[:12])"
  ```

  Rotating `cookie_secret` changes every id at once (and signs everyone out), so match old lines against the secret that was set when they were written.

## Tests

```bash
python3 dev.py pytest
```

The tests are offline and never call AI providers. They include a headless render of the upload app. Before relying on results, try the app on your own batches and compare a few `TEXT_VERIFIED`, `SINGLE_MODEL`, and `RULES_ONLY` rows against the source documents.

## Licence

Copyright (C) 2026 luckyng0221

This project is licensed under the **GNU Affero General Public License v3.0 only** (SPDX identifier `AGPL-3.0-only`); the full text is in [`LICENSE`](LICENSE) at the root of the repository.

**PyMuPDF is the reason.** The PDF work uses PyMuPDF, which is dual-licensed under the AGPL-3.0 or a commercial Artifex licence. The AGPL-3.0 is the free option for this project, so the whole work is released under the AGPL-3.0-only.

**What AGPL section 13 asks of the owner.** Anyone who interacts with a deployed version of this app over a network must be offered the exact Corresponding Source of that version, on the same terms. The app carries its part of this: while `app.source_url` in `config.yaml` is set to a public address, every page — the main page, the sign-in page, and the refusal pages — shows a **Source code (AGPL-3.0)** link to that address. A link to this project's private GitHub repository does not satisfy section 13: nobody who interacts with the deployed app can read a private repository. Publish the deployed version's exact source somewhere public first, then set `app.source_url` in `config.yaml` to that address. Publish a separate public copy of the exact deployed commit (for example a `git archive` of that commit without `docs/` and `.secrets.baseline`, after checking `git ls-files` for any other internal or deployment-specific tracked file). Never make the private repository itself public: that would publish `docs/SECURITY_REVIEW.md`, `docs/ANONYMOUS_ACCESS_PLAN.md`, `.secrets.baseline` and the whole history. While `app.source_url` is empty, the app only shows the line "Licensed under the GNU AGPL-3.0" and makes no link, so making the source offer is the owner's step to complete before opening the app to others.
