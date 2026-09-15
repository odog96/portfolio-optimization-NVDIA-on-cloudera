# Investigation Notes — Model Registry & Model Deployment (2026-09-14/15)

Context for picking this up in a new session. This documents two **separate,
confirmed platform-level issues** in this Cloudera AI workspace, not code
bugs in this repo — plus everything already ruled out, and working
artifacts already in place for when deployment is unblocked.

Environment: CDP environment `applied-ai-cdp-env`, ML Workspace domain
`ml-bba677be-716.applied.jmgjgh.a0.cloudera.site`, project `Portfolio-Optimization`
(project_id `ay03-1kde-9qio-8hel`), AWS account `240534893097`, region `us-east-1`.

## Issue 1: Model Registry — every registration fails on the storage write

**Symptom:** `cmlapi.create_registered_model(...)` always returns a
successful response immediately (status `READY`), but the model version
silently flips to `status: UPLOAD_FAILED` within ~15-20 seconds, with
error `"Failed to create model registry directory in storage: exit status 1"`.

**Confirmed facts:**
- Fails 100% of the time — reproduced 7+ times, different model names,
  different model flavors (lightgbm, pytorch, onnx-pytorch), different
  users (`ozarate` + at least 2 teammates), both new models and new
  versions of existing models.
- The initial API response cannot be trusted — always check status again
  a few seconds later via `get_registered_model()`.
- **Ranger/RAZ audit log shows zero entries** for any of these attempts
  (Service Type = S3, filtered to today) — meaning the failing component
  never even asks Ranger for authorization. Rules out IAM/IDBroker/Ranger
  policy as the cause.
- **The `mlx` Kubernetes namespace (where `model-registry-v1`/`v2`/`db-0`
  run) shows nothing relevant either** — only clean DB-level POST/PATCH
  calls, no mention of S3/storage, no short-lived Job pod appears during
  a failing attempt (checked with live `kubectl` log/event watching,
  twice, with precise timing windows).
- This session's own compute (the CML session that calls the API) also
  shows no local trace (no subprocess, no relevant log file) — so the
  actual storage-write step isn't running client-side either.
- **Conclusion: the failing component is not visible to us anywhere we
  have access** (not `mlx`, not cluster-wide — `kubectl get pods
  --all-namespaces` returns `Forbidden`, not local compute). It's inside
  Cloudera's own control-plane infrastructure (EKS cluster
  `liftie-2psyrznb`, a shared multi-tenant cluster).
- Two "fixes" the user attempted on S3 bucket permissions did not change
  the behavior at all — same error, same timing, both times.

**Registered models with `UPLOAD_FAILED` for reference:** `PortfolioReturnsForecaster`
(`0jbb-g1m0-mesu-qm8f`), `test-forecast`, plus teammates' `nba-risk-pytorch`,
`nba-risk-onnx-pytorch` — all identical error.

**Status: unresolved.** Likely needs Cloudera Support (has full repro
details above), though the user asked to hold off on drafting that ticket
for now — don't push it unprompted.

## Issue 2: CML Model builds stuck at `pending` indefinitely

**Symptom:** Both API-created and UI-created Model builds sit at
`status: pending` and never progress to `building`, for 10+ minutes with
zero movement.

**Confirmed facts:**
- Reproduced with **three separate builds**: a manual UI-created build
  (`ReturnsForecasterServe2`), an API build using our real `serve.py`
  (with dependencies + install script), and a **trivial zero-dependency
  script** (`hello.py`, just adds two numbers, no imports beyond stdlib).
  All three stuck identically.
- Since even the trivial build stalls the same way, this rules out our
  code, dependencies, file paths, and resource profile choice as the
  cause — **this looks like a scheduling-level stall for the whole
  workspace**, separate from Issue 1.
- Resource group config looks healthy on paper (`resource_profile_id=4`
  → "Default CPU Group", 16 CPU, autoscale min=1/max=10, `allow_models=True`)
  — not artificially capped at zero, though we can't see actual current
  utilization/node scheduling from here.

**Status: unresolved.**

## Key gotchas learned (useful regardless of the above)

- **CML "project root" for `file_path` in `CreateModelBuildRequest` is
  `/home/cdsw`, not the git repo directory.** Use
  `file_path='portfolio-optimization/serve.py'`, not `'serve.py'` — a
  bare filename fails with `"script 'serve.py' not found in project
  directory"`. (Confirmed by inspecting the UI-created build's own
  `file_path` field.)
- **`resource_profile_id` is required on `CreateModelBuildRequest`** —
  omitting it causes an *instant* `build failed` (not `pending`), a
  different failure mode than the stall above. `resource_profile_id=4`
  = 2 CPU / 4GB on the Default CPU Group.
- A fresh build container has nothing pip-installed beyond the base
  runtime image — use `build_script_path` pointing at a script that
  does `pip install -r requirements.txt && pip install -e .` (see
  `cml_build.sh`) if the serving script imports this package.
- **CML self-tests a Model's predict function with an empty/placeholder
  payload during build** — the function must handle missing/malformed
  input gracefully (return an error dict) rather than raise, or the
  build fails.
- **Feature engineering minimum data requirement**: `compute_features()`
  in `src/forecasting/feature_engineering.py` hardcodes a 63-day SMA
  feature and always trims `iloc[63:]` regardless of config — so **any
  caller needs 64+ trading days of price history per ticker**, not the
  ~25-30 days one might assume from the configurable rolling windows.
  This is a pre-existing quirk in the forecasting code, not something
  introduced during this investigation.

## Working artifacts already in the project (untracked in git as of writing)

- **`predict.py`** — CML Model entry point, `predict_with_metrics(args)`,
  loads `models/returns_forecaster.lgb`, takes `{"prices": {ticker:
  {date: price}}}`, returns `{"predicted_returns": {ticker: float}}`.
- **`serve.py`** — same contract as `predict.py`, but uses the proper
  `@cdsw.model_metrics` decorator (with a local no-op fallback so it can
  be smoke-tested outside a real CML container) and tracks
  `n_tickers`/`mean_predicted_return` metrics. **This is the one to
  actually deploy** once builds are unblocked.
- **`hello.py`** — trivial `add_numbers(args)` smoke test, no
  dependencies. Useful for re-testing whether basic Model builds work
  again in a fresh session.
- **`cml_build.sh`** — build-time install script (`pip install -r
  requirements.txt && pip install -e .`).
- **`CLAUDE.md`** — updated with forecasting module architecture and the
  Cloudera AI Workbench deployment path (`.project-metadata.yaml`,
  `scripts/`), separate from the README's Docker/uv instructions.

None of the above are committed to git — that's deliberate, per the
user's instruction not to commit yet.

## Verified serve.py/predict.py signature (for manual UI testing)

Input:
```json
{"prices": {"<TICKER>": {"<date>": <price>, ...}, ...}}
```
- Minimum 64+ trading days of history per ticker (see gotcha above).

Output, success:
```json
{"predicted_returns": {"<TICKER>": <float>, ...}}
```
Output, bad/insufficient input:
```json
{"predicted_returns": {}, "error": "<description>"}
```

A verified-working sample payload (3 tickers, real dow30 data) was
generated during this session at `/tmp/sample_payload.json` — regenerate
similarly via `portfolio_optimization.utils.get_input_data(...).tail(100)`
if needed, since `/tmp` won't persist to a new session.

## Recommended next steps

1. Decide whether to open a Cloudera Support case for Issues 1 and 2 —
   both are well-documented above with exact repro steps and are outside
   what's fixable from inside the project.
2. If/when builds work again, deploy `serve.py` (not `predict.py` —
   it has the correct `cdsw.model_metrics` decorator) and test against
   the payload format above.
3. The Ranger/CM access fix (Knox security group `sgr-...`, adding the
   user's VPN IP) is already done and confirmed working — unrelated to
   the two issues above, no further action needed there.
