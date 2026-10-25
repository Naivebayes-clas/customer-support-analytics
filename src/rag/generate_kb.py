"""
Generate RavenStack product help documentation (simulated knowledge base)
+ Q&A pairs for RAG evaluation.

Run: python -m src.rag.generate_kb
Creates:
  data/knowledge_base_ravenstack/*.txt  (25 help articles)
  data/qa_pairs_ravenstack.json         (30 Q&A pairs)
"""
import json
import os
from pathlib import Path

KB_DIR = "./data/knowledge_base_ravenstack"
os.makedirs(KB_DIR, exist_ok=True)

# ─── KNOWLEDGE BASE DOCUMENTS ─────────────────────────────────────────────────

docs = {}

# --- BILLING CLUSTER (5 docs, strong distractors) ---

docs["billing_cycles"] = """
RavenStack Billing Cycles

Your RavenStack subscription renews on the same day each month that you first
subscribed. For example, if you subscribed on March 15th, your billing cycle
renews on the 15th of every subsequent month.

Billing cycle length is always 30 days for monthly plans and 365 days for
annual plans. Annual plans are billed in a single payment at the start of the
cycle and include a 20% discount compared to monthly billing.

Proration: If you upgrade or downgrade mid-cycle, RavenStack calculates the
unused portion of your current plan and applies a credit or charge to your
next invoice. Prorated amounts appear as a separate line item on your invoice.

Grace period: If a payment fails, your account enters a 7-day grace period.
During this period, all features remain active. If payment is not resolved
within 7 days, the account is downgraded to the free tier and data is
retained for 30 days before permanent deletion.

You can view your next billing date, current cycle status, and payment history
in Settings → Billing → Overview.
"""

docs["plan_comparison"] = """
RavenStack Plans: Basic, Pro, Enterprise

Basic Plan ($49/user/month):
- Up to 10 team members
- 5 GB storage per workspace
- Standard support (48-hour response SLA)
- Core API access (100 requests/minute)
- 3 integrations
- Community forum access

Pro Plan ($99/user/month):
- Up to 50 team members
- 50 GB storage per workspace
- Priority support (12-hour response SLA)
- Extended API access (1,000 requests/minute)
- Unlimited integrations
- Advanced analytics dashboard
- Custom report exports (CSV, PDF)
- SSO (SAML 2.0)

Enterprise Plan (custom pricing):
- Unlimited team members
- Unlimited storage
- Dedicated support (4-hour response SLA, named CSM)
- Unlimited API access (custom rate limits)
- On-premises or dedicated cloud deployment
- Custom SLA agreements
- Audit log retention (13 months)
- Security review and pen-test reports
- Dedicated onboarding (2 weeks)
- Quarterly business reviews

All plans include: unlimited projects, unlimited API tokens, data export,
and 99.9% uptime SLA (Enterprise: 99.99%).

To upgrade or downgrade, navigate to Settings → Billing → Change Plan.
Changes take effect immediately with prorated billing.
"""

docs["managing_subscriptions"] = """
Managing Your RavenStack Subscription

Canceling your subscription:
Go to Settings → Billing → Cancel Plan. Your subscription remains active
until the end of the current billing cycle. After the cycle ends, your
account transitions to the free tier. You retain read-only access to your
data for 30 days. After 30 days, all data is permanently deleted and cannot
be recovered.

You cannot cancel mid-cycle. If you need to stop billing immediately, contact
support@ravenstack.io and we will issue a prorated refund for the unused
portion of your cycle.

Changing your billing frequency:
Monthly ↔ Annual: Go to Settings → Billing → Billing Frequency. Switching
from monthly to annual takes effect on your next renewal. Switching from
annual to monthly takes effect immediately (you receive a prorated credit for
the remaining annual period).

Updating payment method:
Go to Settings → Billing → Payment Methods. You can add, remove, or set a
default payment method. We support Visa, Mastercard, Amex, and ACH (for
Enterprise annual plans).

Invoicing:
PDF invoices are generated automatically at the start of each billing cycle.
Access them in Settings → Billing → Invoices. You can also request a custom
invoice with your company's tax ID or PO number by contacting billing.
"""

docs["refund_policy"] = """
RavenStack Refund Policy

Standard refunds:
RavenStack offers a 14-day money-back guarantee on all new subscriptions.
If you are not satisfied within the first 14 days of your subscription,
contact support with your account ID and we will issue a full refund to your
original payment method within 5 business days.

Prorated refunds (cancellation):
If you cancel during your billing cycle and request a refund, we calculate
the unused portion of your cycle and issue a credit. The credit is applied
to your account balance first. If the credit exceeds your balance, the
remainder is refunded to your payment method within 5-7 business days.

Enterprise refunds:
Enterprise contracts are governed by the terms in your signed agreement.
Standard refund policy does not apply to Enterprise accounts. Contact your
Customer Success Manager for contract-specific refund terms.

Non-refundable items:
- Custom development work (API integrations, custom reports)
- Onboarding fees (Enterprise)
- Data migration services
- Training sessions

To request a refund: Settings → Billing → Request Refund, or email
billing@ravenstack.io with your account ID and reason.
"""

docs["payment_methods"] = """
Accepted Payment Methods and Billing

Credit/Debit Cards:
We accept Visa, Mastercard, and American Express for all plan tiers.
Card information is stored securely via Stripe (PCI-DSS Level 1 compliant).
We never store full card numbers on our servers.

ACH (Enterprise, annual plans only):
Enterprise customers on annual billing can pay via ACH transfer. To set up
ACH, contact billing@ravenstack.io with your bank routing number, account
number, and authorized signer name. ACH payments take 3-5 business days to
process. Your subscription activates once the payment clears.

Purchase orders (Enterprise):
Enterprise customers can request invoice-based billing with net-30 or net-60
terms. A valid purchase order must be submitted before the first billing
cycle. POs are processed within 2 business days.

Currency:
All plans are priced in USD. International customers are charged in USD;
your bank may apply foreign transaction fees. We do not offer multi-currency
billing at this time.

Failed payments:
If a payment fails, we retry automatically on days 1, 3, and 7. After the
third failed attempt, your account enters the grace period (see Billing
Cycles documentation).
"""

# --- API CLUSTER (5 docs, strong distractors) ---

docs["api_authentication"] = """
RavenStack API Authentication

All API requests require authentication via a Bearer token in the
Authorization header.

Obtaining a token:
1. Navigate to Settings → API → Tokens
2. Click "Generate New Token"
3. Name your token (e.g., "CI/CD Pipeline")
4. Select scopes: read, write, admin
5. Click "Generate" — the token is displayed once. Store it securely.

Token format:
Authorization: Bearer rsk_live_a1b2c3d4e5f6...

Token types:
- Live tokens (rsk_live_...): Full access to production data
- Test tokens (rsk_test_...): Sandbox environment, no real data
- Webhook signing secret: Used to verify webhook payloads (not for API auth)

Token rotation:
We recommend rotating tokens every 90 days. Old tokens remain valid for 24
hours after a new token is generated (grace period). To revoke a token
immediately, delete it from Settings → API → Tokens.

IP allowlisting (Enterprise):
Enterprise plans can restrict API access to specific IP ranges. Configure
in Settings → API → Security → IP Allowlist.

Rate limits apply per token. See API Rate Limits documentation for details.
"""

docs["api_rate_limits"] = """
RavenStack API Rate Limits

Rate limits are enforced per API token, per minute.

Basic Plan: 100 requests/minute
Pro Plan: 1,000 requests/minute
Enterprise: Custom (default 5,000/minute; negotiable)

When you exceed the rate limit, the API returns HTTP 429 (Too Many Requests)
with a Retry-After header indicating how many seconds to wait.

Headers returned with every response:
- X-RateLimit-Limit: Your plan's rate limit
- X-RateLimit-Remaining: Requests remaining in current window
- X-RateLimit-Reset: Unix timestamp when the window resets
- Retry-After: (only on 429) Seconds to wait before retrying

Best practices:
1. Cache responses for idempotent GET requests
2. Use exponential backoff with jitter for 429 responses
3. Batch operations where possible (e.g., create 100 items in one call)
4. Monitor X-RateLimit-Remaining in your application logging

Burst allowance:
You can burst up to 2× your rate limit for a maximum of 10 seconds per
minute. This is useful for batch imports. Sustained rates above your plan
limit will be throttled.

Bulk operations:
For imports exceeding 10,000 records, use the Bulk API endpoint
(/v2/bulk/import) which has separate, higher rate limits and processes
asynchronously.
"""

docs["api_webhooks"] = """
RavenStack Webhooks

Webhooks allow your application to receive real-time notifications when
events occur in your RavenStack workspace.

Setting up a webhook:
1. Navigate to Settings → API → Webhooks
2. Click "Add Endpoint"
3. Enter your HTTPS URL (must be publicly accessible)
4. Select event types: ticket.created, ticket.resolved, user.invited,
   billing.invoice_paid, project.archived, etc.
5. Click "Save" — your signing secret is displayed once

Payload format:
{
  "id": "evt_abc123",
  "type": "ticket.resolved",
  "created_at": "2024-03-15T10:30:00Z",
  "data": {
    "ticket_id": "T-0024de",
    "account_id": "A-712f1c",
    "resolution_time_hours": 4.2,
    "satisfaction_score": 4
  }
}

Signature verification:
Each request includes a X-RavenStack-Signature header containing an
HMAC-SHA256 signature of the raw request body, keyed with your signing
secret. Always verify this signature before processing the payload.

Retry policy:
If your endpoint returns a non-2xx response or times out (10-second limit),
we retry with exponential backoff: 1 min, 5 min, 30 min, 2 hours, 12 hours.
After 5 failed attempts (spanning ~24 hours), the webhook is marked as
failed and no further retries are attempted. You can view delivery history
in Settings → API → Webhooks → [endpoint] → Logs.

Webhook ordering:
Webhooks are delivered in the order events occur, but concurrent delivery
is not guaranteed. If ordering matters, use the created_at timestamp to
re-sequence events in your application.
"""

docs["api_versioning"] = """
RavenStack API Versioning

The RavenStack API uses URL-based versioning. The current stable version is
v2. Previous versions (v1) are in maintenance mode and will be deprecated
on January 1, 2027.

Base URL:
- Production: https://api.ravenstack.io/v2
- Sandbox: https://api.ravenstack.io/v2/sandbox

Version lifecycle:
1. Beta: New endpoints are released under /v2/beta/ for 60 days
2. GA: Endpoints graduate to /v2/ after beta period
3. Deprecation: We send 90 days' notice via email and dashboard banner
4. Sunset: Deprecated endpoints return 410 Gone after the sunset date

Breaking changes:
We do not make breaking changes to GA endpoints within a major version.
Breaking changes (removing fields, changing types, renaming endpoints)
only occur in major version bumps (v2 → v3).

Non-breaking changes:
Adding new fields to responses, adding new endpoints, adding new enum
values to existing fields — these are non-breaking and may appear without
notice. Always handle unknown fields gracefully in your code.

Migration guide:
When a new major version is released, we publish a migration guide at
docs.ravenstack.io/api/migration with a field-by-field mapping.
"""

docs["api_error_handling"] = """
RavenStack API Error Handling

All errors return a JSON body with a machine-readable code and a
human-readable message:

{
  "error": {
    "code": "validation_error",
    "message": "Field 'email' must be a valid email address.",
    "field": "email",
    "request_id": "req_abc123"
  }
}

HTTP status codes:
- 400 Bad Request: Malformed request body or invalid parameters
- 401 Unauthorized: Missing or invalid API token
- 403 Forbidden: Token is valid but lacks required scope
- 404 Not Found: Resource does not exist (or you lack access)
- 409 Conflict: Resource state prevents the operation (e.g., deleting an
  active subscription)
- 422 Unprocessable Entity: Request is well-formed but semantically invalid
- 429 Too Many Requests: Rate limit exceeded (see Rate Limits docs)
- 500 Internal Server Error: Something went wrong on our end
- 503 Service Unavailable: Temporary outage (check status.ravenstack.io)

Idempotency:
For POST and PUT requests, include an Idempotency-Key header (UUID) to
prevent duplicate operations on retry. The key is valid for 24 hours.
If you retry with the same key, we return the original response without
re-processing.

Debugging:
Include the request_id from error responses when contacting support.
We can trace any request by its ID within 30 days.
"""

# --- SECURITY CLUSTER (4 docs) ---

docs["sso_configuration"] = """
Single Sign-On (SSO) Configuration

RavenStack supports SAML 2.0 SSO for Pro and Enterprise plans.

Prerequisites:
- Pro or Enterprise plan
- Admin role in RavenStack
- SAML IdP configured (Okta, Azure AD, Google Workspace, OneLogin, etc.)

Setup steps:
1. Navigate to Settings → Security → SSO
2. Click "Configure SAML"
3. Enter your IdP metadata URL (or paste XML)
4. Set the ACS (Assertion Consumer Service) URL:
   https://app.ravenstack.io/saml/acs
5. Set the Entity ID: https://app.ravenstack.io
6. Map attributes: email → user email, firstName → display name
7. Click "Activate"

Once active, all users in your workspace must authenticate via SSO.
Password login is disabled. To add a new user, they must be provisioned
in your IdP first (SCIM or manual).

SCIM provisioning:
Enterprise plans support SCIM 2.0 for automated user provisioning and
deprovisioning. Configure the SCIM endpoint in your IdP:
- Endpoint: https://api.ravenstack.io/v2/scim
- Bearer token: Generate in Settings → Security → SCIM

Enforcement:
Once SSO is enabled, you can enforce it (blocking password login) or run
in "optional" mode where users can choose SSO or password.
"""

docs["two_factor_auth"] = """
Two-Factor Authentication (2FA)

RavenStack supports TOTP-based 2FA (Google Authenticator, Authy, 1Password)
and hardware security keys (FIDO2/WebAuthn).

Enabling 2FA:
1. Navigate to Settings → Security → Two-Factor Authentication
2. Select method: TOTP or Security Key
3. For TOTP: Scan the QR code with your authenticator app, enter the 6-digit
   code to confirm
4. For Security Key: Touch the key when prompted
5. Generate and store backup codes (10 single-use codes)

Enforcement:
Admins can require 2FA for all workspace members. Navigate to
Settings → Security → Policies → Require 2FA. Once enabled, users who
haven't set up 2FA will be prompted on their next login and cannot access
the workspace until they complete setup.

Recovery:
If you lose access to your 2FA device:
1. Use a backup code (single-use)
2. Contact workspace admin to reset your 2FA
3. Enterprise: CSM can verify identity and reset

2FA does not apply to API tokens. API authentication is handled by token
scoping and IP allowlisting (see API Authentication docs).
"""

docs["audit_logs"] = """
Audit Logs

RavenStack maintains an audit log of all significant actions in your
workspace. Pro plans retain logs for 6 months; Enterprise plans retain
logs for 13 months.

Viewing logs:
Navigate to Settings → Security → Audit Logs. Filter by:
- User (who performed the action)
- Action type (login, permission_change, data_export, etc.)
- Date range
- Resource (which project, user, or setting was affected)

Export:
Pro: CSV export (last 30 days)
Enterprise: CSV or JSON export (full retention period), or stream to your
SIEM via the Audit Log API endpoint (/v2/audit-logs).

Logged events include:
- Authentication: login, logout, 2FA challenge, password reset
- Authorization: role change, permission grant/revoke, SSO configuration
- Data: record created/updated/deleted, bulk export, API access
- Billing: plan change, payment method update, invoice generated
- Admin: workspace settings change, user invited/suspended/removed

Each log entry includes: timestamp (UTC), user ID, IP address, action,
resource ID, and a before/after state diff for configuration changes.

Audit logs are immutable — they cannot be edited or deleted by any user,
including workspace owners.
"""

docs["data_retention"] = """
Data Retention and Deletion

Active data:
While your subscription is active, all data (projects, tickets, files,
audit logs) is retained indefinitely. There is no automatic data expiration
for active accounts.

After cancellation:
- Free tier: Data retained for 30 days, then permanently deleted
- Paid plans (canceled): Data retained for 30 days post-cancellation
- Enterprise: Data retained per your contract terms (minimum 90 days)

Data deletion request (GDPR/CCPA):
You can request permanent deletion of your data at any time:
1. Navigate to Settings → Security → Data Deletion
2. Or email privacy@ravenstack.io with your account ID
3. Deletion is processed within 30 days
4. You receive a confirmation email with a deletion certificate

What is deleted:
- All workspace data (projects, tickets, files, conversations)
- User profiles and authentication credentials
- Audit logs (for the requesting user)
- API tokens and webhook configurations

What is retained (legal requirements):
- Billing records (7 years, per tax law)
- Communication records related to active disputes
- Aggregated/anonymized usage statistics

Data residency:
All data is stored in US-East (Virginia) by default. Enterprise customers
can request EU (Frankfurt) or APAC (Singapore) residency at onboarding.
Data does not move between regions after initial provisioning.
"""

# --- SUPPORT CLUSTER (4 docs) ---

docs["ticket_priorities"] = """
Support Ticket Priorities and SLAs

When you submit a support ticket, it is assigned a priority based on
impact and urgency. You can request a priority at submission time; our
support team may adjust it after triage.

Priority levels:

P1 - Critical:
- Production outage or data loss
- Security vulnerability
- SLA: 1-hour response, 4-hour resolution target
- Available: All plans (P1 auto-escalates to engineering)

P2 - High:
- Major feature broken, no workaround
- Performance degradation >50%
- SLA: 4-hour response, 24-hour resolution target
- Available: All plans

P3 - Medium:
- Feature bug with workaround
- Configuration question
- SLA: 12-hour response, 72-hour resolution target (Basic: 48h)
- Available: All plans

P4 - Low:
- Feature request
- Documentation improvement
- Cosmetic issue
- SLA: 24-hour response, no resolution target
- Available: Pro and Enterprise (Basic: community forum)

Escalation:
Any ticket can be escalated by the support agent if the resolution
time exceeds the SLA target. Escalated tickets are routed to the
engineering on-call or a senior support specialist.
"""

docs["escalation_policy"] = """
Support Escalation Policy

Automatic escalation:
Tickets are automatically escalated when:
1. Response SLA is breached (no agent response within the priority window)
2. Resolution SLA is breached (ticket open beyond target)
3. Customer requests escalation (click "Escalate" in the ticket)
4. Ticket is re-opened within 48 hours of resolution

Escalation routing:
- P1 → Engineering on-call (immediate page via PagerDuty)
- P2 → Senior Support Specialist (next available)
- P3 → Team Lead (end of current shift)
- P4 → Product team backlog (weekly review)

What happens on escalation:
1. The ticket is tagged with "escalated" and the reason
2. The new assignee receives a notification with full context
3. The customer receives an email confirming the escalation
4. A resolution timestamp is reset (the new assignee gets a fresh SLA)
5. The escalation is logged in the audit trail

De-escalation:
If the escalated issue is resolved by the original agent (e.g., the
escalation was a false alarm), the ticket can be de-escalated. This
does not affect the customer's SLA clock.

Escalation does not guarantee a faster resolution — it guarantees a
higher-level owner is assigned. For P1 issues, escalation triggers
an incident bridge (Slack channel + phone bridge for Enterprise).
"""

docs["onboarding_process"] = """
Customer Onboarding Process

Basic and Pro plans:
Self-serve onboarding. After signup, you receive:
1. A welcome email with a 15-minute product tour link
2. In-app guided setup (3 steps: invite team, create project, connect
   first integration)
3. Access to the help center and community forum

No dedicated onboarding session is included. If you need help, submit a
P3 ticket and our support team will walk you through setup.

Enterprise plan:
Dedicated 2-week onboarding program with a named Customer Success
Manager (CSM). The program includes:

Week 1:
- Day 1: Kickoff call (30 min) — goals, timeline, stakeholder mapping
- Day 2-3: SSO configuration, SCIM provisioning, API setup
- Day 4-5: Data migration (we import your existing data), team training
  (2 × 1-hour live sessions)

Week 2:
- Day 1-2: Integration setup (webhooks, custom reports, dashboards)
- Day 3: Go-live support (CSM on standby)
- Day 4: Feedback session, success metrics agreement
- Day 5: Onboarding complete → transition to steady-state CSM

Post-onboarding:
- Monthly check-in calls (first 90 days)
- Quarterly Business Reviews (QBRs)
- Annual security review and pen-test report delivery

Onboarding fees:
Enterprise onboarding is included in the annual contract. Custom
onboarding (beyond the standard 2-week program) is billed at $250/hour.
"""

docs["support_hours"] = """
Support Hours and Channels

Response times by plan:

Basic:
- Email: 48-hour response (business hours: 9am-6pm ET, Mon-Fri)
- Community forum: 72-hour response
- No phone or chat support

Pro:
- Email: 12-hour response (24/7 for P1/P2, business hours for P3/P4)
- Live chat: 9am-6pm ET, Mon-Fri (average wait: 5 minutes)
- Phone: Not included

Enterprise:
- Email: 4-hour response (24/7 for all priorities)
- Live chat: 24/7 (average wait: 2 minutes)
- Phone: 24/7 dedicated line
- Slack Connect: Shared channel with support engineering
- Named CSM: Direct access, 1-hour response

Holidays:
During US holidays (Thanksgiving, Christmas, New Year's Day), response
times may be extended by 24 hours. P1 issues are always handled 24/7
regardless of holidays.

Submitting a ticket:
1. Click the ? icon in the bottom-right corner of the app
2. Or email support@ravenstack.io (include your account ID)
3. Or use the API: POST /v2/tickets

Tracking:
All tickets are visible in Settings → Support → My Tickets. You can
add internal notes (visible only to your team) or public replies
(visible to the support agent).
"""

# --- ADMIN CLUSTER (3 docs) ---

docs["user_management"] = """
User Management and Roles

RavenStack supports four roles with increasing permissions:

Viewer:
- View projects, tickets, and reports
- Cannot create, edit, or delete anything
- Cannot access billing or settings

Member:
- Create and edit projects, tickets, and documents
- Invite other Viewers
- Access API (read + write scopes)
- Cannot manage billing, users, or workspace settings

Admin:
- All Member permissions
- Manage users (invite, suspend, remove, change roles)
- Configure workspace settings (name, timezone, SSO, 2FA policy)
- Access audit logs
- Cannot manage billing

Owner:
- All Admin permissions
- Manage billing (change plan, update payment, cancel)
- Delete workspace (irreversible)
- Transfer ownership to another user

Role changes:
Navigate to Settings → Team → [user] → Change Role. Role changes take
effect immediately. A user cannot change their own role from Owner to
anything else unless another Owner exists.

Inviting users:
1. Settings → Team → Invite
2. Enter email, select role
3. User receives an email with a 7-day acceptance link
4. If SSO is enforced, the user must be provisioned in your IdP first

Suspended users:
Suspended users lose all access immediately but their data (created
tickets, comments) is preserved. You can re-activate a suspended user
at any time.
"""

docs["workspace_settings"] = """
Workspace Settings

General:
- Workspace name: Visible to all members and in API responses
- Timezone: Affects SLA calculations and report timestamps
- Default language: UI language for new members (en, es, fr, de, ja)
- Week start: Monday or Sunday (affects calendar views and reports)

Security:
- SSO: Enable/disable SAML SSO (see SSO Configuration docs)
- 2FA policy: Optional or Required for all members
- IP allowlist: Restrict access to specific IP ranges (Enterprise)
- Session timeout: 15 min, 30 min, 1 hour, 4 hours, 8 hours
- Password policy: Minimum length, complexity requirements

Notifications:
- Email: Per-user preference (all, mentions only, none)
- Slack: Connect a workspace for real-time ticket notifications
- Webhooks: See API Webhooks documentation
- Digest: Daily or weekly summary email of ticket activity

Data:
- Default project visibility: Private (only invited members) or
  Public (all workspace members can view)
- File upload limit: 10 MB (Basic), 100 MB (Pro), 1 GB (Enterprise)
- Retention: See Data Retention documentation

Danger zone:
- Export all data: Generates a ZIP with all projects, tickets, files
  (available to Admin and Owner)
- Delete workspace: Irreversible. Requires Owner role and typed
  confirmation of workspace name. Data is purged within 72 hours.
"""

docs["integrations"] = """
RavenStack Integrations

RavenStack integrates with 50+ tools via native connectors and the
public API.

Native integrations (one-click setup):
- Slack: Real-time ticket notifications, /ravenstack commands
- Jira: Bidirectional sync (tickets ↔ issues)
- GitHub: Link tickets to PRs and issues
- Salesforce: Sync account and contact records
- HubSpot: Sync deal and company data
- Zapier: 5,000+ app connections (Pro and Enterprise)
- Microsoft Teams: Ticket notifications and inline views

Setup:
1. Navigate to Settings → Integrations
2. Find the tool you want to connect
3. Click "Connect" and follow the OAuth flow
4. Configure sync settings (what to sync, direction, frequency)

API-based integrations:
For tools without a native connector, use the RavenStack API:
- REST API: https://api.ravenstack.io/v2
- Webhooks: Real-time event delivery (see Webhooks docs)
- Bulk API: For large imports/exports (see Rate Limits docs)

Custom integrations (Enterprise):
Enterprise customers can request a custom integration built by our
solutions engineering team. Typical timeline: 2-4 weeks. Priced based
on scope (starting at $5,000 for simple webhook bridges, $25,000+ for
complex bidirectional syncs).

Integration health:
Each integration shows a status indicator (green/yellow/red) in the
Integrations page. Red indicates a sync failure — check the integration
logs for details.
"""

# ─── WRITE FILES ──────────────────────────────────────────────────────────────

for name, content in docs.items():
    path = Path(KB_DIR) / f"{name}.txt"
    path.write_text(content.strip(), encoding="utf-8")

print(f"✓ Wrote {len(docs)} documents to {KB_DIR}/")

# ─── Q&A PAIRS ────────────────────────────────────────────────────────────────

qa_pairs = [
    # BILLING CLUSTER (distractors: billing_cycles, plan_comparison, managing_subscriptions, refund_policy, payment_methods)
    {
        "query": "What happens if my credit card payment fails?",
        "gold_answer": "We retry on days 1, 3, and 7. After three failed attempts, your account enters a 7-day grace period where features remain active. If unresolved, the account is downgraded to the free tier.",
        "expected_doc": "payment_methods"
    },
    {
        "query": "How does prorated billing work when I upgrade mid-cycle?",
        "gold_answer": "RavenStack calculates the unused portion of your current plan and applies a credit or charge to your next invoice as a separate line item.",
        "expected_doc": "billing_cycles"
    },
    {
        "query": "Can I cancel my subscription immediately and get a refund?",
        "gold_answer": "You cannot cancel mid-cycle normally, but you can contact support for a prorated refund of the unused portion. The credit is applied to your balance first, with any remainder refunded within 5-7 business days.",
        "expected_doc": "managing_subscriptions"
    },
    {
        "query": "What is the difference between Pro and Enterprise plans?",
        "gold_answer": "Pro includes 50 team members, 50 GB storage, 12-hour SLA, 1000 req/min API, and SSO. Enterprise adds unlimited members/storage, 4-hour SLA, custom rate limits, dedicated CSM, audit log retention, and on-premises deployment.",
        "expected_doc": "plan_comparison"
    },
    {
        "query": "How long is the money-back guarantee?",
        "gold_answer": "14 days from the start of your subscription. Contact support with your account ID for a full refund to your original payment method within 5 business days.",
        "expected_doc": "refund_policy"
    },

    # API CLUSTER (distractors: api_authentication, api_rate_limits, api_webhooks, api_versioning, api_error_handling)
    {
        "query": "How do I verify that a webhook payload is legitimate?",
        "gold_answer": "Each request includes a X-RavenStack-Signature header with an HMAC-SHA256 signature of the raw body, keyed with your signing secret. Verify this signature before processing.",
        "expected_doc": "api_webhooks"
    },
    {
        "query": "What happens when I exceed the API rate limit?",
        "gold_answer": "The API returns HTTP 429 with a Retry-After header. You can burst up to 2× your limit for 10 seconds per minute. Use exponential backoff with jitter for retries.",
        "expected_doc": "api_rate_limits"
    },
    {
        "query": "How do I prevent duplicate operations when retrying an API call?",
        "gold_answer": "Include an Idempotency-Key header (UUID) with POST and PUT requests. The key is valid for 24 hours; retries with the same key return the original response without re-processing.",
        "expected_doc": "api_error_handling"
    },
    {
        "query": "When will API v1 be deprecated?",
        "gold_answer": "API v1 will be deprecated on January 1, 2027. We send 90 days' notice before sunset, and a migration guide is published at docs.ravenstack.io/api/migration.",
        "expected_doc": "api_versioning"
    },
    {
        "query": "How do I rotate my API token safely?",
        "gold_answer": "Generate a new token in Settings → API → Tokens. The old token remains valid for 24 hours (grace period). To revoke immediately, delete the old token. We recommend rotating every 90 days.",
        "expected_doc": "api_authentication"
    },

    # SECURITY CLUSTER (distractors: sso_configuration, two_factor_auth, audit_logs, data_retention)
    {
        "query": "How do I set up SAML SSO with Okta?",
        "gold_answer": "Go to Settings → Security → SSO → Configure SAML. Enter your Okta metadata URL, set the ACS URL to https://app.ravenstack.io/saml/acs, map email and firstName attributes, then activate.",
        "expected_doc": "sso_configuration"
    },
    {
        "query": "What happens to my data if I cancel my subscription?",
        "gold_answer": "Data is retained for 30 days after cancellation. After 30 days, all data is permanently deleted. Enterprise accounts retain data per contract terms (minimum 90 days).",
        "expected_doc": "data_retention"
    },
    {
        "query": "How long are audit logs retained?",
        "gold_answer": "Pro plans retain audit logs for 6 months; Enterprise plans retain them for 13 months. Logs are immutable and cannot be edited or deleted by any user.",
        "expected_doc": "audit_logs"
    },
    {
        "query": "What should I do if I lose access to my 2FA device?",
        "gold_answer": "Use a backup code (single-use), contact your workspace admin to reset your 2FA, or for Enterprise, your CSM can verify identity and reset.",
        "expected_doc": "two_factor_auth"
    },

    # SUPPORT CLUSTER (distractors: ticket_priorities, escalation_policy, onboarding_process, support_hours)
    {
        "query": "What is the response time for a P1 ticket on the Basic plan?",
        "gold_answer": "1-hour response and 4-hour resolution target. P1 is available on all plans and auto-escalates to engineering.",
        "expected_doc": "ticket_priorities"
    },
    {
        "query": "What triggers an automatic escalation of a support ticket?",
        "gold_answer": "Automatic escalation triggers when: the response SLA is breached, the resolution SLA is breached, the customer requests it, or the ticket is re-opened within 48 hours of resolution.",
        "expected_doc": "escalation_policy"
    },
    {
        "query": "What does Enterprise onboarding include?",
        "gold_answer": "A dedicated 2-week program with a named CSM: kickoff call, SSO/SCIM configuration, data migration, 2 training sessions, integration setup, go-live support, and a feedback session with success metrics.",
        "expected_doc": "onboarding_process"
    },
    {
        "query": "What support channels are available on the Pro plan?",
        "gold_answer": "Email (12-hour response, 24/7 for P1/P2) and live chat (9am-6pm ET, Mon-Fri, ~5 min wait). Phone support is not included on Pro.",
        "expected_doc": "support_hours"
    },

    # ADMIN CLUSTER (distractors: user_management, workspace_settings, integrations)
    {
        "query": "What is the difference between Admin and Owner roles?",
        "gold_answer": "Admins can manage users, workspace settings, and audit logs but cannot manage billing. Owners add billing management, workspace deletion, and ownership transfer.",
        "expected_doc": "user_management"
    },
    {
        "query": "How do I connect RavenStack to Slack?",
        "gold_answer": "Go to Settings → Integrations → Slack → Connect, then follow the OAuth flow. You get real-time ticket notifications and /ravenstack commands in your channels.",
        "expected_doc": "integrations"
    },
    {
        "query": "How do I change my workspace timezone?",
        "gold_answer": "Navigate to Settings → Workspace → General → Timezone. This affects SLA calculations and report timestamps for all members.",
        "expected_doc": "workspace_settings"
    },
    {
        "query": "Can I restrict API access to specific IP addresses?",
        "gold_answer": "Yes, Enterprise plans can configure IP allowlisting in Settings → API → Security → IP Allowlist. All API requests from non-allowlisted IPs are rejected.",
        "expected_doc": "api_authentication"
    },
    {
        "query": "What is the file upload limit on the Pro plan?",
        "gold_answer": "100 MB per file. Basic is 10 MB, Enterprise is 1 GB. The limit is configured in workspace settings.",
        "expected_doc": "workspace_settings"
    },
    {
        "query": "How does SCIM provisioning work for automated user management?",
        "gold_answer": "Enterprise plans support SCIM 2.0. Configure the endpoint (https://api.ravenstack.io/v2/scim) in your IdP with a Bearer token from Settings → Security → SCIM. Users are auto-provisioned and deprovisioned.",
        "expected_doc": "sso_configuration"
    },
    {
        "query": "What is the webhook retry policy if my endpoint is down?",
        "gold_answer": "We retry with exponential backoff: 1 min, 5 min, 30 min, 2 hours, 12 hours. After 5 failed attempts (~24 hours total), the webhook is marked as failed with no further retries.",
        "expected_doc": "api_webhooks"
    },
    {
        "query": "Can I export all my data before canceling?",
        "gold_answer": "Yes. Go to Settings → Workspace → Danger Zone → Export All Data. This generates a ZIP with all projects, tickets, and files. Available to Admin and Owner roles.",
        "expected_doc": "workspace_settings"
    },
]

# Write QA pairs
qa_path = "./data/qa_pairs_ravenstack.json"
with open(qa_path, "w") as f:
    json.dump(qa_pairs, f, indent=2)

print(f"✓ Wrote {len(qa_pairs)} Q&A pairs to {qa_path}")
print(f"\nDistractor clusters:")
print(f"  Billing:   {len([d for d in docs if 'billing' in d or 'plan' in d or 'subscription' in d or 'refund' in d or 'payment' in d])} docs")
print(f"  API:       {len([d for d in docs if d.startswith('api_')])} docs")
print(f"  Security:  {len([d for d in docs if d in ('sso_configuration','two_factor_auth','audit_logs','data_retention')])} docs")
print(f"  Support:   {len([d for d in docs if d in ('ticket_priorities','escalation_policy','onboarding_process','support_hours')])} docs")
print(f"  Admin:     {len([d for d in docs if d in ('user_management','workspace_settings','integrations')])} docs")   
