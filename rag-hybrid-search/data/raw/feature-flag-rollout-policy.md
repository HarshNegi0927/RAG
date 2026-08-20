# Feature Flag Rollout Policy

## Naming Convention
Flags are named `<team>.<feature>.<short-description>`, e.g. `billing.usage-alerts.enabled`. Flags without a team prefix are rejected by the flag service at creation time.

## Standard Rollout Stages
New behavior behind a flag should move through: internal-only (Loopstack employees), 1% of customers, 10%, 50%, 100%. Each stage should run for at least 24 hours before advancing, longer for anything touching billing or the ledger. Advancing stages requires the feature owner to explicitly confirm no error-rate or latency regression in the rollout dashboard — advancement is not automatic.

## Kill Switch
Every flag can be force-disabled instantly via `POST /internal/flags/kill` with the flag name, which sets it to 0% for all users regardless of its configured rollout and requires no deploy. This is the fastest lever available during an incident if a new feature is suspected as the cause — pulling the kill switch is always safe to do first and ask questions second.

## Flag Debt
Flags that have been at 100% for more than 60 days should be removed from the codebase — the flag check itself becomes dead weight and a source of confusion. The flag service sends an automated Slack reminder to the flag owner at the 60 day mark, and again at 90 days if it's still not cleaned up.

## Who Can Change What
Any engineer can create a flag and advance it through internal and 1% stages. Advancing past 10% for anything touching payments or the ledger requires a second engineer's sign-off, logged as a comment on the rollout ticket.
