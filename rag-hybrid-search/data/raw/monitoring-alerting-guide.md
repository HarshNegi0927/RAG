# Monitoring and Alerting Guide

## Metrics Stack
Services export metrics via Prometheus client libraries; Grafana is the primary visualization layer. The main operational dashboard is **"Loopstack Golden Signals"**, which every service is required to have a row on, covering latency, traffic, errors, and saturation (the standard four golden signals).

## Standard Alert Rules
The most common production alert is `HighErrorRate5xx`, which fires when a service's 5xx rate exceeds 2% over a 5 minute window, sustained for at least 3 consecutive evaluations to avoid flapping on transient blips. `HighP99Latency` fires when p99 latency exceeds the service's configured SLO threshold (default 1000ms unless overridden per service) for 10 minutes.

## Alert Routing
Alerts tagged `severity=page` go to PagerDuty and wake someone up. Alerts tagged `severity=ticket` create a Jira ticket during business hours and are not paged. Every new alert rule must specify a severity tag explicitly — alerts without one are rejected by the alerting pipeline's validation step at commit time.

## Dashboards Per Service
Every service is expected to maintain its own Grafana dashboard beyond the shared Golden Signals view, covering anything specific to that service's failure modes — queue depth for `notification-service`, migration duration for anything using `loopmigrate`, Kafka consumer lag for anything reading from `payment.events.v2`.

## On-Call Dashboard Checklist
When responding to a page, the first three things to check are: the Golden Signals dashboard for the affected service, recent deploys in the `#deploys` Slack channel (most incidents follow a recent change), and the dependency graph in the service catalog to see if the real root cause is upstream.
