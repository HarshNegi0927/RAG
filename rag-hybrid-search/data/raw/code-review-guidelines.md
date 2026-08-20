# Code Review Guidelines

## Approval Requirements
Every pull request needs at least 1 approval before merging. PRs touching `billing-service`, `ledger-service`, or any code path that moves money require 2 approvals, at least one from a designated Payments reviewer (listed in `CODEOWNERS`).

## What Reviewers Should Check
Correctness matters more than style — style is handled by tooling, not humans. Reviewers should focus on: does this do what the PR description says, are there tests for the new behavior, are error paths handled, and is there anything here that would be hard for the next person to understand in six months. Nitpicks are welcome but should be marked as non-blocking (`nit:`) so they don't hold up merges.

## Linting and Formatting
All Python code is formatted and linted with `ruff`, run automatically as a pre-commit hook and again in CI. CI will block merges on lint failures; it will not block on `ruff`'s style suggestions marked as warnings, only on errors. There is no manual style debate in review — if `ruff` doesn't flag it, it's not a blocking issue.

## PR Size
Aim for PRs under 400 lines of diff. Larger changes should be split into a stack of smaller PRs where possible, each independently reviewable. If a large PR is genuinely unavoidable (a big migration, a vendored dependency update), say so explicitly in the description so reviewers calibrate their expectations.

## Response Time Expectations
Reviewers are expected to give a first pass within one business day. If you're blocked waiting on review for longer than that, it's fine to ping the reviewer directly or ask in the team channel — this is not considered rude.

## Merging
Authors merge their own PRs once approvals are in and CI is green; reviewers do not merge on behalf of authors. Squash-merge is the default; use a merge commit only when preserving individual commit history is genuinely useful (rare).
