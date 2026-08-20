# Deployment Guide

## Overview
Loopstack services are deployed through the internal CLI tool `loopctl`, which wraps our GitHub Actions pipelines and talks to the Kubernetes clusters in each region. All production deploys must go through the pipeline — direct `kubectl apply` to prod is disabled for everyone except the on-call SRE during an active incident.

## Standard Deploy Flow
1. Merge your PR to `main` after the required approvals (see Code Review Guidelines).
2. CI builds the image, runs the unit and integration suite, and pushes to the internal registry tagged with the short commit SHA.
3. Deploy to staging automatically. Staging soak time is 20 minutes minimum before prod is unlocked.
4. Run `loopctl deploy --env=prod --service=<name>` to promote the staging image to production. This requires the `deployer` role in Okta.
5. The deploy is rolled out canary-style: 5% of pods for 10 minutes, then 50%, then 100%, unless you pass `--fast` (only allowed for hotfixes tagged `sev1-fix`).

## Rollbacks
`loopctl rollback --env=prod --service=<name>` reverts to the last known-good image within seconds, since it just repoints the deployment to the previous tag. Rollback does not require additional approval and can be run by anyone with the `deployer` role. Always rollback first and investigate second if a canary is failing error-rate checks.

## Freeze Windows
No production deploys are allowed Friday 4pm through Monday 9am local time, except SEV1/SEV2 hotfixes. The freeze is enforced by the pipeline itself — `loopctl deploy` will refuse to run and print `ERR_FREEZE_WINDOW` outside of business hours unless the `--override-freeze` flag is used, which requires a VP sign-off logged in the deploy ticket.

## Environment Promotion
We run four environments: `dev`, `staging`, `perf`, and `prod`. Config differences between environments live in `config/<env>.yaml` in each service repo — never hardcode environment-specific values in application code. The `perf` environment mirrors prod sizing and is used for load testing before major launches.
