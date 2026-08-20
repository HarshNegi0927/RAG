# Data Retention and Privacy Policy

## Data Classification Tiers
- **Tier 1 (Restricted)**: payment card data, government ID numbers. Never stored directly — handled entirely by our PCI-compliant payment processor.
- **Tier 2 (Confidential)**: customer PII — names, emails, billing addresses, usage data.
- **Tier 3 (Internal)**: business metrics, aggregated analytics, non-customer operational data.
- **Tier 4 (Public)**: marketing content, published documentation.

## Retention Periods
When a customer closes their account, Tier 2 data is retained for 90 days to allow for account recovery in case of accidental closure, then permanently deleted. Application logs containing request metadata are retained for 30 days. Database backups are retained for 400 days for disaster recovery purposes, but backups are not searchable or queryable outside of a full restore — they are not a mechanism for retrieving individually deleted records.

## Deletion Requests
A customer-initiated deletion request (GDPR "right to erasure" or equivalent) triggers immediate deletion of Tier 2 data from all primary datastores, overriding the standard 90-day grace period. Deletion requests are processed by the `customer-service` team's data deletion workflow and must be completed within 30 days per policy, though in practice primary-store deletion happens within 72 hours; the 30-day window mostly covers purging from backups and any downstream analytics copies.

## Access Logging
All access to Tier 1 and Tier 2 data by internal engineers is logged, including read-only queries run through the internal admin tools. These access logs are themselves retained for 400 days and are reviewed quarterly by the security team.

## Third-Party Processors
Any new vendor that will touch Tier 2 data requires a Data Processing Agreement reviewed by Legal before integration work begins, not after. This applies to analytics tools, customer support platforms, and any AI/ML vendor, without exception.
