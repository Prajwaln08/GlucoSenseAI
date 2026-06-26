"""
GlucoSense AI — AI Coach.

Hybrid, safety-first (per the design in README.md):
  - context.py — read-only snapshot of the user's state (the FACTS).
  - rules.py   — deterministic safety flags + lifestyle recommendations (numbers
                 computed in code; the LLM never invents them).
  - tools.py   — log_food / log_vitals tool-use so the coach can record from chat.
  - llm.py     — Claude client (server-side key only) with a graceful no-key fallback.
  - service.py — orchestration + persistence (ChatMessage / Recommendation).

Mounted at /coach in the API. The mobile app holds NO key — it calls the backend.
"""
