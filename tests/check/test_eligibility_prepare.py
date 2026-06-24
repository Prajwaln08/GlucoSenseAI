"""
Phase 7 — eligibility, cohorts, and the prepare orchestrator.

Synthetic profiles exercise the gating/cohort logic; one guarded real run takes
two CGMacros users through Steps 1 → 4.

Run:
    conda activate glucosenseai
    pytest tests/check/test_eligibility_prepare.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.step3_imputation import STRUCTURAL_SIGNALS
from src.data import eligibility as el
from src.data import prepare as prep
from src.data.step1_loader import discover_users


def _profile_frame(uid, dataset, days, hr_cov=1.0, avail=("hr",)):
    n = days * 144                                  # 144 ten-min rows per day
    idx = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    hr = np.full(n, 70.0)
    if hr_cov < 1.0:
        hr[int(n * hr_cov):] = np.nan
    df = pd.DataFrame({"glucose_mg_dl": 100.0, "hr": hr}, index=idx)
    df["uid"] = uid
    df["dataset"] = dataset
    for s in STRUCTURAL_SIGNALS:
        df[f"{s}_available"] = 1 if s in avail else 0
    return df


# ── Profiling ─────────────────────────────────────────────────────────────────

def test_profile_counts_cgm_days_and_watch_coverage():
    p = el.profile_user(_profile_frame("cg-017", "cgmacros", days=14, hr_cov=1.0))
    assert p.cgm_days == 14
    assert p.watch_coverage == pytest.approx(1.0)
    assert "hr" in p.availability


# ── Gating ────────────────────────────────────────────────────────────────────

def test_post_cgm_requires_14_days():
    short = el.profile_user(_profile_frame("cg-017", "cgmacros", days=12))
    ok    = el.profile_user(_profile_frame("cg-018", "cgmacros", days=14))
    assert not el.is_eligible(short, "post_cgm")
    assert el.is_eligible(ok, "post_cgm")


def test_while_on_cgm_needs_only_8_total_days():
    p8 = el.profile_user(_profile_frame("cg-019", "cgmacros", days=8))
    p7 = el.profile_user(_profile_frame("cg-021", "cgmacros", days=7))
    assert el.is_eligible(p8, "while_on_cgm")       # 6 train + 2 val, no test
    assert not el.is_eligible(p7, "while_on_cgm")   # < 8 days
    assert not el.is_eligible(p8, "post_cgm")       # needs 14


def test_watch_compulsory():
    p = el.profile_user(_profile_frame("cg-020", "cgmacros", days=14, hr_cov=0.1))
    assert not el.is_eligible(p, "post_cgm")        # 10% < 30% coverage


def test_reserved_users_excluded():
    p = el.profile_user(_profile_frame("cg-001", "cgmacros", days=20))
    assert not el.is_eligible(p, "post_cgm")        # cg-001 is a demo user
    assert el.is_eligible(p, "post_cgm", allow_reserved=True)


# ── Cohorts by availability ───────────────────────────────────────────────────

def test_cohorts_group_by_availability_signature():
    frames = pd.concat([
        _profile_frame("cg-031", "cgmacros", 14, avail=("hr", "mets", "calories_burned")),
        _profile_frame("cg-032", "cgmacros", 14, avail=("hr", "mets", "calories_burned")),
        _profile_frame("np-003", "nature_paper", 14, avail=("hr", "eda", "temp")),
    ])
    profiles = el.profile_table(frames)
    groups = el.cohorts(profiles, "post_cgm")
    assert len(groups) == 2                          # cgmacros-like vs np-like
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 2]


def test_cohorts_skip_degenerate_small_cohorts():
    frames = pd.concat([
        _profile_frame("cg-031", "cgmacros", 14, avail=("hr", "mets", "calories_burned")),
        _profile_frame("cg-032", "cgmacros", 14, avail=("hr", "mets", "calories_burned")),
        _profile_frame("cg-033", "cgmacros", 14, avail=("hr", "mets", "calories_burned")),
        _profile_frame("np-003", "nature_paper", 14, avail=("hr", "eda", "temp")),  # lone 1-user cohort
    ])
    profiles = el.profile_table(frames)
    groups = el.cohorts(profiles, "post_cgm", min_users=3)
    assert len(groups) == 1                          # the 1-user np cohort is dropped
    assert list(groups.values())[0] == ["cg-031", "cg-032", "cg-033"]


# ── Real prepare (guarded) ────────────────────────────────────────────────────

def test_prepare_two_cgmacros_users_end_to_end():
    ids = discover_users("cgmacros")
    if len(ids) < 2:
        pytest.skip("Need ≥2 CGMacros users on disk.")
    out = prep.prepare(mode="cgm_active", users={"cgmacros": ids[:2]})
    assert out.table["uid"].nunique() == 2
    assert "target_delta_30" in out.table.columns
    assert len(out.profiles) == 2
    # both users have HR → eligible-or-reserved decision is computable
    for uid, p in out.profiles.items():
        assert p.cgm_days > 0
