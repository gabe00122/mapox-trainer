---
name: merge
description: >-
  Land finished jaxrl (Mapox Trainer) changes on main: commit, rebase, run the
  applicable local checks, and fast-forward main locally and on GitHub. Use
  only when the user explicitly requests landing, not for review, checks alone,
  or skill installation.
disable-model-invocation: true
metadata:
  delta-action: land
---

# Merge into jaxrl `main`

An explicit `/merge` invocation or Delta **Merge** action authorizes landing
the requested jaxrl change. In a thread with multiple projects, work only in
the jaxrl worktree. Land by fast-forward; do not create a merge commit, force
push, or bypass GitHub branch protections.

## Destinations

Confirm that `origin` points to the jaxrl repository on GitHub before pushing.
In a managed Delta checkout with a `local` backlink, update the primary
checkout's `main` first, then GitHub's `main`. The primary checkout must be on
`main` with no uncommitted changes. If it is dirty or Git refuses the update,
stop rather than stashing changes, changing its configuration, or pushing only
to GitHub without the user's approval. With no `local` backlink, skip the
`local` fetch, push, and verification.

## Steps

1. Fetch `origin` and, if present, `local`. If `local/main` and `origin/main`
   have diverged, stop and ask. Otherwise use the tip that contains the other
   as the base. Check which commits each push would publish; do not include
   unrelated local-only commits.
2. Stage and commit only the work the user asked to land. Follow the recent
   commit style: an imperative subject and a short body when useful. Ask if
   unrelated changes cannot be separated safely.
3. Rebase the unpublished change onto the base if needed. Resolve conflicts
   only when the intended result is clear; otherwise abort the rebase and ask.
4. Run the checks below on the rebased commit. Do not push if a required check
   fails or cannot run.
5. With a `local` backlink, recheck the primary checkout and run
   `git push local HEAD:refs/heads/main`. Verify that the primary checkout's
   `HEAD` equals the checked commit. Then run
   `git push origin HEAD:refs/heads/main`.
6. Verify the published commit with `git ls-remote origin refs/heads/main`.
   Report the commit and the checks actually run. If only one destination was
   updated, name it and say what blocked the other. Use `report-landed` for
   the final result in Delta.

## Checks

- For every change, run `git diff --check <base> HEAD`, replacing `<base>`
  with the selected base commit.
- For changes to trainer code, tests, scripts, training configs, or Python
  dependencies and toolchain files (including `pyproject.toml`, `uv.lock`,
  `requirements.txt`, and `.python-version`), run
  `JAX_PLATFORMS=cpu uv run --locked pytest` from the repository root.
  `uv` uses the project's Python version and installs the dev group with
  pytest. The CPU suite does not require CUDA or TPU extras.
- Run a focused smoke check when the change affects a workflow that pytest
  does not exercise. Documentation-only and skill-only changes do not
  require the Python suite.
