# Incident Response Runbook

## Severity Levels
- **SEV1** — Full outage or data loss risk affecting all customers. Must be acknowledged within 5 minutes and requires an Incident Commander (IC) within 10 minutes.
- **SEV2** — Major functionality degraded for a large subset of customers. Acknowledge within 15 minutes.
- **SEV3** — Minor degradation, workaround available. Acknowledge within 2 hours during business hours.
- **SEV4** — Cosmetic or low-impact issue. No page; file a ticket.

## Getting Paged
On-call rotations are managed in PagerDuty under the `loopstack-oncall-primary` and `loopstack-oncall-secondary` schedules. If you're paged and it's the middle of the night and you're not sure what's happening, the first move is always the same: acknowledge the page, join the `#incidents` Slack channel, and post "IC needed" if none has claimed the incident yet. You do not need to have diagnosed anything before joining the call.

## Incident Commander Responsibilities
The IC does not necessarily fix the issue themselves. Their job is to: keep a running timeline in the incident doc, make sure the right specialists are pulled in, decide when to escalate severity, and own customer communication via the status page. The IC role rotates independently of who's on-call — anyone trained as an IC can pick it up.

## Escalation Path
If the primary on-call doesn't acknowledge within the SEV1/SEV2 window, PagerDuty automatically pages the secondary, then the engineering manager, then the VP of Engineering. Never wait silently for someone else to notice — re-page if you're unsure whether the right people are engaged.

## Postmortems
Every SEV1 and SEV2 requires a blameless postmortem published within 5 business days. Postmortems live in the `incidents` repo and must include a timeline, root cause, and at least two follow-up action items with owners and due dates. SEV3s get a postmortem only if requested by the service owner.
