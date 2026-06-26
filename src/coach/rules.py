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
    """Lifestyle nudges (diet/activity) grounded in the snapshot. Each: {kind,title,body}."""
    g = ctx.get("glucose")
    pred = ctx.get("predicted")
    act = ctx.get("today_activity")
    foods = ctx.get("recent_food") or []
    recs: list[dict] = []

    if g is None:
        return [{
            "kind": "activity", "title": "Connect your glucose data",
            "body": "Sync a CGM through Health Connect to get personalised, real-time coaching here.",
        }]

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

    if not recs:
        recs.append({
            "kind": "diet", "title": "Nice and steady",
            "body": f"Glucose is holding {g['trend']} around {g['latest']} mg/dL with "
                    f"{g['time_in_range_pct']}% time-in-range. Keep your current rhythm going.",
        })
    return recs[:3]
