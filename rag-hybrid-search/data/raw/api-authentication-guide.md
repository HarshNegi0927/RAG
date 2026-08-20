# API Authentication Guide

## Overview
All external API requests to Loopstack must include the `X-Loopstack-Auth` header containing a signed API key. Requests without this header, or with an expired key, receive a `401` with error body `ERR_AUTH_MISSING` or `ERR_AUTH_EXPIRED` respectively.

## Key Rotation
API keys are rotated using the internal `rotate_api_key()` function exposed via the `auth-service` gRPC interface, or via `loopctl auth rotate-key --customer=<id>` for support engineers. Rotating a key does not immediately invalidate the old one — there is a 24 hour grace period during which both keys work, to avoid breaking customers mid-request. After the grace period, the old key returns `ERR_AUTH_EXPIRED`.

## OAuth Flow
For customer-facing integrations, we support OAuth 2.0 authorization code flow. The token endpoint is `/oauth/token`, and access tokens are short-lived JWTs valid for 1 hour. Refresh tokens are valid for 30 days and are revoked automatically if unused for 14 consecutive days.

## Rate Limits
Each API key has a default rate limit of 100 requests per minute, returned in the `X-RateLimit-Remaining` response header. Exceeding the limit returns `429` with error code `ERR_RATE_LIMITED`. Enterprise customers can request higher limits through the account team; this is configured per-key in the `auth-service` admin panel, not via a customer-facing self-serve flow.

## Common Mistakes
The most frequent support ticket in this area is customers signing requests with a staging key against the production endpoint, which fails silently-looking but actually returns `ERR_AUTH_INVALID_ENV`. Always check the key prefix: `lpk_live_` for production, `lpk_test_` for staging/sandbox.
