# Service Architecture Overview

## High-Level Shape
Loopstack is a set of independently deployed services communicating primarily over gRPC for synchronous calls and Kafka for asynchronous events. There is no shared database between services — each service owns its own schema, and cross-service data access goes through the owning service's API, never direct database access.

## Core Services
- **auth-service** — issues and validates API keys and OAuth tokens, owns the `rotate_api_key()` interface.
- **billing-service** — subscription plans, invoicing, and usage-based billing calculations.
- **ledger-service** — the system of record for all financial transactions; append-only, never updates rows in place.
- **notification-service** — outbound email, SMS, and webhook delivery, with retry and dead-letter handling.
- **customer-service** (internal name, unrelated to support) — account and organization metadata, team membership.
- **ingestion-service** — receives customer usage events and writes them to the analytics pipeline.

## Event Backbone
Services publish domain events to Kafka topics namespaced by service and version, for example `payment.events.v2` and `account.created.v1`. Consumers should always be written to tolerate at-least-once delivery — duplicate events are possible and every consumer must be idempotent. Topic schemas are registered in the internal schema registry and are backwards-compatible by policy; breaking changes require a new versioned topic, not a mutation of an existing one.

## Data Flow Example: A Payment
When a customer is charged, `billing-service` computes the amount and calls `ledger-service` synchronously to record the transaction. `ledger-service` then emits a `payment.events.v2` event. `notification-service` consumes that event to send a receipt email, and `ingestion-service` consumes it separately to feed the analytics warehouse. Note that the receipt email is not sent synchronously as part of the charge — if `notification-service` is down, the charge still succeeds and the receipt is delivered once the consumer catches up.

## Ownership
Every service has a single owning team listed in `service-catalog.yaml`. Paging the wrong team for a service issue is the single most common cause of slow SEV1 response times — check the catalog before escalating if you're not sure who owns something.
