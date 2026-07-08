"""
Deterministic safety + lifestyle logic. These functions compute the FACTS in
plain Python — the LLM never produces these numbers, it only explains them.
"""

from __future__ import annotations


def red_flags(ctx: dict) -> list[str]:
    """Urgent, non-negotiable safety messages from the latest glucose value."""
    g = ctx.get("glucose")
    if not g:
        return []
    latest = g["latest"]
    flags: list[str] = []
    if latest < 54:
        flags.append(f"Last reading is very low ({latest} mg/dL): treat with ~15 g fast-acting "
                     "carbs now, recheck in 15 minutes, and seek urgent help if you don't improve.")
    elif latest < 70:
        flags.append(f"Last reading is low ({latest} mg/dL): have ~15 g fast carbs and recheck in 15 minutes.")
    elif latest > 300:
        flags.append(f"Last reading is very high ({latest} mg/dL): check ketones if you can and "
                     "contact your care team — don't make insulin changes on your own.")
    elif latest > 250:
        flags.append(f"Last reading is high ({latest} mg/dL): hydrate, hold off on more carbs, and recheck soon.")
    return flags


def recommendations(ctx: dict) -> list[dict]:
    """Doctor Gluco's suggestion ladder — priority-ordered, always grounded in the
    user's own numbers, never generic filler. Each: {kind,title,body}.

      T1 safety (current or PREDICTED low/high)  →  always wins the top slot
      T2 forecast-reactive coaching
      T3 data-gap nudges (log food — only when a gap is detected)
      T4 pattern insight / steady-state positive reinforcement
    """
    from datetime import datetime, timezone

    g = ctx.get("glucose")
    pred = ctx.get("predicted")
    act = ctx.get("today_activity")
    foods = ctx.get("recent_food") or []
    recs: list[dict] = []

    # ── T1: safety — current red flags first, then predicted lows/highs ──────
    for flag in red_flags(ctx)[:1]:
        recs.append({"kind": "safety", "title": "Check your glucose now", "body": flag})
    if g and pred and not recs:
        if pred["mgdl"] < 75 <= g["latest"]:
            recs.append({
                "kind": "safety", "title": "Heading low — eat something small",
                "body": f"You're at {g['latest']} mg/dL but forecast near {round(pred['mgdl'])} "
                        f"in {pred['horizon_min']} min. A small snack now heads that off.",
            })
        elif pred["mgdl"] > 250 and g["latest"] <= 250:
            recs.append({
                "kind": "safety", "title": "A high is coming",
                "body": f"Forecast is near {round(pred['mgdl'])} mg/dL in {pred['horizon_min']} min. "
                        "Hydrate, skip extra carbs, and keep an eye on it.",
            })

    # ── No CGM: Virtual-CGM users get an estimate-quality nudge, not filler ──
    if g is None:
        if act or foods:
            return [{
                "kind": "diet", "title": "Your logs power your estimate",
                "body": "Virtual CGM leans on your meals and watch data. Log what you eat "
                        "(just tell Doctor Gluco in chat) and your estimates get sharper.",
            }]
        return [{
            "kind": "activity", "title": "Connect your glucose data",
            "body": "Sync a CGM — or connect your watch for Virtual CGM estimates — to get "
                    "personalised, real-time coaching here.",
        }]

    # ── T3: gap detection — glucose moved but no meal on record ─────────────
    last_food_h = None
    if foods and foods[0].get("logged_at"):
        try:
            ts = datetime.fromisoformat(str(foods[0]["logged_at"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            last_food_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        except ValueError:
            pass
    if g["trend"] == "rising" and (last_food_h is None or last_food_h > 4):
        recs.append({
            "kind": "diet", "title": "Rising — what did you eat?",
            "body": f"Your glucose is climbing (now {g['latest']} mg/dL) but there's no recent "
                    "meal on record. Tell Doctor Gluco what you ate so your forecast learns from it.",
        })

    # Predicted rise → movement blunts the spike
    if pred and (pred["mgdl"] - g["latest"] > 25 or pred["mgdl"] > 180):
        recs.append({
            "kind": "activity", "title": "A short walk now helps",
            "body": f"Your glucose is forecast near {round(pred['mgdl'])} mg/dL in {pred['horizon_min']} min. "
                    "A 10–15 minute walk can soften that rise.",
        })

    # Carb-heavy recent meal + rising → meal-pairing tip
    high_carb = next((f for f in foods if (f.get("carbs_g") or 0) > 45), None)
    if high_carb and g["trend"] == "rising":
        recs.append({
            "kind": "diet", "title": "Balance carb-heavy meals",
            "body": "Your last meal was carb-rich. Pairing carbs with protein, fibre or healthy fat "
                    "next time can flatten the post-meal rise.",
        })

    # Low time-in-range
    if g["time_in_range_pct"] < 60:
        recs.append({
            "kind": "diet", "title": "Lift your time-in-range",
            "body": f"You've been in range about {g['time_in_range_pct']}% of the last few hours. "
                    "Steadier, lower-GI meals can help nudge that up.",
        })

    # Low activity
    if act and act.get("steps") is not None and act["steps"] < 5000:
        recs.append({
            "kind": "activity", "title": "Add a little movement",
            "body": f"About {act['steps']} steps so far today. A short walk gets you closer to a steadier day.",
        })

    # ── T4: steady state → specific positive reinforcement, never filler ────
    if not recs:
        recs.append({
            "kind": "diet", "title": "Nice and steady",
            "body": f"Glucose is holding {g['trend']} around {g['latest']} mg/dL with "
                    f"{g['time_in_range_pct']}% time-in-range. Keep your current rhythm going.",
        })
    return recs[:3]
