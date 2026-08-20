# Database Migration Guide

## Tooling
Schema migrations run through `loopmigrate`, a thin wrapper around Alembic that adds our safety checks. Migrations live in `db/migrations/` in each service repo and are numbered sequentially.

## Running a Migration
`loopmigrate up` applies all pending migrations. In production this is invoked automatically as a pre-deploy step by the pipeline, not run manually, except for backfills. The migration step will fail the deploy if it does not complete within the timeout configured by the environment variable `DB_MIGRATION_TIMEOUT_SEC`, which defaults to 900 seconds (15 minutes) in production and 120 seconds in staging.

## Connection Pooling
Each service connects to Postgres through PgBouncer. The maximum pool size per service instance is controlled by `DB_POOL_MAX_CONN`, defaulting to 50. During a migration, `loopmigrate` temporarily drops the effective pool to 10 connections to avoid lock contention on large tables, then restores the original value once the migration completes.

## Large Table Migrations
For tables over 10 million rows, do not run a blocking `ALTER TABLE` directly. Use the online schema change pattern: create a shadow table, backfill in batches of 5,000 rows using the `loopmigrate backfill` command, then swap. This avoids holding a long lock that would block writes on high-traffic tables like `payments` or `ledger_entries`.

## Rollback
Every migration must have a corresponding `down` migration that has actually been tested in staging, not just written. `loopmigrate down` reverts the most recent migration. If a migration has already backfilled data, the down migration must handle that data explicitly rather than assuming an empty table.

## Review Requirements
Any migration touching a table larger than 1 million rows, or touching `payments`, `ledger_entries`, or `user_accounts`, requires sign-off from a member of the Data Platform team in addition to the normal code review.
