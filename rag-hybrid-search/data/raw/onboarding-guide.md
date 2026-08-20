# New Engineer Onboarding Guide

## Before Day One
IT will ship your laptop pre-imaged with the standard engineering setup. You'll get an email with your Okta invite — accept it before your start date if you can, since most of our internal tools (GitHub, PagerDuty, the `loopctl` CLI, AWS console access) are gated behind Okta SSO.

## Week One
Day 1 is orientation and account setup. By day 3, you should have made your first pull request — it does not need to be meaningful. Most teams keep a list of small "good first issue" tickets specifically for this. Your onboarding buddy (assigned by your manager) is your first point of contact for anything you're unsure about; there are no dumb questions in the first two weeks.

## Access Requests
Production database read access, AWS console access beyond your own team's resources, and access to customer data all require a request through the `access-requests` Slack workflow, which routes to your manager and the relevant system owner for approval. Most requests are approved within one business day. Do not share credentials or access tokens between engineers, even temporarily.

## Required Reading
Before touching production code, read: the Deployment Guide, the Incident Response Runbook, and the Code Review Guidelines. You are not expected to memorize any of it — just know it exists and roughly what it covers so you know where to look later.

## Local Development
Run `make bootstrap` in any service repo to set up your local environment, which spins up dependent services via Docker Compose and seeds a local database with anonymized sample data. If `make bootstrap` fails, the most common cause is Docker not being allocated enough memory — 8GB minimum is recommended in Docker Desktop settings.

## Your First On-Call Shift
Engineers are not added to the on-call rotation until they've been at Loopstack for at least 6 weeks and have shadowed at least two full on-call shifts.
