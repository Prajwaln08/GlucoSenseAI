"""
Claude client for the coach. Turns the structured snapshot + the user's message
into warm, safe, plain-language coaching. Degrades gracefully with no API key.
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src import config
from src.coach import rules
from src.coach.tools import TOOLS, execute_tool
from src.db.models import User
from src.utils import get_logger

log = get_logger(__name__)

DISCLAIMER = ("I'm an educational coach, not a doctor — I can't diagnose or change "
              "your medication or insulin. Please check with your care team for those.")

SYSTEM = f"""You are GlucoSense, a warm, encouraging diabetes & lifestyle coach for one user.

You are given a CONTEXT snapshot with the only real numbers you may cite. Never invent
glucose values, predictions, or stats that aren't in the snapshot. If you don't have a
number, say so plainly.

Safety (non-negotiable):
- You are NOT a clinician. Never diagnose, never give insulin or medication doses, never
  tell the user to change a prescription. Defer those to their care team.
- If the snapshot lists urgent SAFETY FLAGS, lead with them clearly and calmly.
- For anything that sounds like an emergency, tell them to seek medical help now.

Logging:
- When the user mentions eating something, call log_food. When they report a blood
  pressure, weight, finger-stick glucose, or HbA1c, call log_vitals. Then confirm briefly.

Style: supportive and practical, plain language, 2–4 short sentences. No markdown headers.
{DISCLAIMER}"""


def _client():
    if not config.ANTHROPIC_API_KEY:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _system(ctx: dict) -> str:
    flags = rules.red_flags(ctx)
    parts = [SYSTEM, "\n\nCONTEXT (the only numbers you may cite):\n" + json.dumps(ctx, default=str)]
    if flags:
        parts.append("\n\nURGENT SAFETY FLAGS:\n- " + "\n- ".join(flags))
    return "".join(parts)


def chat(db: Session, user: User, ctx: dict, history: list[dict], message: str) -> tuple[str, list[dict]]:
    """Return (reply_text, actions). actions = tool side-effects performed."""
    client = _client()
    if client is None:
        return _fallback(ctx, message), []

    messages = [*history, {"role": "user", "content": message}]
    actions: list[dict] = []
    try:
        for _ in range(4):  # cap tool-use turns
            resp = client.messages.create(
                model=config.COACH_MODEL, max_tokens=config.COACH_MAX_TOKENS,
                system=_system(ctx), tools=TOOLS, messages=messages,
            )
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        summary, action = execute_tool(db, user, block.name, dict(block.input))
                        actions.append(action)
                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": summary})
                messages.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return text.strip() or DISCLAIMER, actions
        return "Let's keep going — what would you like to do next?", actions
    except Exception as exc:                               # noqa: BLE001
        log.warning(f"coach chat failed: {exc}")
        return _fallback(ctx, message), actions


def _fallback(ctx: dict, message: str) -> str:
    """Safe, useful reply when the LLM is unavailable (no key / error)."""
    flags = rules.red_flags(ctx)
    if flags:
        return flags[0] + " " + DISCLAIMER
    g = ctx.get("glucose")
    if g:
        return (f"Your glucose is sitting {g['trend']} around {g['latest']} mg/dL "
                f"({g['time_in_range_pct']}% in range lately). I can log meals and vitals for you, "
                f"and once the AI coach is fully enabled I'll give richer guidance. {DISCLAIMER}")
    return ("I'm here to help with food, activity and your glucose trends. Connect a CGM to get "
            f"personalised coaching. {DISCLAIMER}")
