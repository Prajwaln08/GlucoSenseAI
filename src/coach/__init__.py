"""
GlucoSense AI — AI Coach (RESERVED FOR A FUTURE PHASE — NOT BUILT YET).

This package is an intentional STUB. Per the project direction (directive #4),
the AI chat/coach interface that gives food & activity suggestions over the
user's own data is DEFERRED. Nothing here is wired into the API.

When the AI phase begins (see docs/implementation_plan.md §7 / Phase 6):
  - it will read EXISTING tables read-only (CgmReading / WearableActivity /
    FoodLog) via the same query layer the web UI uses,
  - no LLM dependency, router, or endpoint is added until then.

See README.md in this directory for the deferred design.
"""
