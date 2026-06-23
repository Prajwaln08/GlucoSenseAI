"""
Step 1 — Loader checks.

Covers the unified-schema contract and the cross-dataset id scheme without
depending on the heavy raw data: synthetic frames exercise the union/NaN logic,
and one guarded real CGMacros load confirms the parser wiring still works.

Run:
    conda activate glucosenseai
    pytest tests/check/test_step1_loader.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.step1_loader import (
    META_COLS,
    SIGNAL_SOURCE,
    UNIFIED_COLUMNS,
    LoadedUser,
    build_unified_table,
    discover_users,
    load_user,
    make_uid,
    unify_columns,
)


# ── Identity ──────────────────────────────────────────────────────────────────

def test_make_uid_namespaces_both_datasets():
    assert make_uid("nature_paper", "003") == "np-003"
    assert make_uid("cgmacros", "017") == "cg-017"


def test_make_uid_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        make_uid("fitbit_study", "001")


def test_uids_do_not_collide_across_datasets():
    # Same raw id in both datasets must yield distinct uids.
    assert make_uid("nature_paper", "001") != make_uid("cgmacros", "001")


# ── Schema integrity ──────────────────────────────────────────────────────────

def test_unified_columns_have_no_duplicates():
    assert len(UNIFIED_COLUMNS) == len(set(UNIFIED_COLUMNS))


def test_every_unified_column_has_a_source():
    for col in UNIFIED_COLUMNS:
        assert col in SIGNAL_SOURCE, f"{col} missing from SIGNAL_SOURCE"
        assert SIGNAL_SOURCE[col] in ("both", "nature_paper", "cgmacros")


def test_shared_target_present_in_schema():
    assert "glucose_mg_dl" in UNIFIED_COLUMNS
    assert SIGNAL_SOURCE["glucose_mg_dl"] == "both"


# ── unify_columns: union of columns, NaN where absent ─────────────────────────

def _fake_cgmacros_frame(uid="cg-017", n=5) -> pd.DataFrame:
    """A CGMacros-shaped wide frame: shared + cgmacros-only columns, no NP signals."""
    idx = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "glucose_mg_dl": np.linspace(100, 120, n),
            "hr": np.linspace(70, 75, n),
            "total_carb": 0.0,
            "mets": 1.0,
            "calories_burned": 0.5,
            "age": 40, "bmi": 25.0,
            # a raw helper column that is NOT in the schema -> should be dropped
            "image_path": "x.png",
        },
        index=idx,
    )
    df["uid"] = uid
    df["dataset"] = "cgmacros"
    df["participant_id"] = uid.split("-", 1)[1]
    return df


def test_unify_columns_exact_schema_and_order():
    out = unify_columns(_fake_cgmacros_frame())
    assert list(out.columns) == META_COLS + UNIFIED_COLUMNS


def test_unify_columns_keeps_present_drops_extras_nans_absent():
    out = unify_columns(_fake_cgmacros_frame())
    # present signal preserved
    assert out["glucose_mg_dl"].notna().all()
    assert (out["mets"] == 1.0).all()
    # NP-only signal absent for a CGMacros user -> structural NaN
    assert out["eda"].isna().all()
    assert out["bvp"].isna().all()
    assert out["acc_magnitude_mean"].isna().all()
    # non-schema helper column dropped
    assert "image_path" not in out.columns
    # identity preserved
    assert (out["uid"] == "cg-017").all()
    assert (out["dataset"] == "cgmacros").all()


# ── build_unified_table: single table for both datasets ───────────────────────

def test_build_unified_table_concats_wide_users_and_defers_sources():
    u1 = LoadedUser("cg-017", "cgmacros", "017", frame=_fake_cgmacros_frame("cg-017"))
    u2 = LoadedUser("cg-018", "cgmacros", "018", frame=_fake_cgmacros_frame("cg-018"))
    # An NP user (no wide frame yet — read in Step 2) must be deferred, not crash.
    u3 = LoadedUser("np-003", "nature_paper", "003")

    table = build_unified_table([u1, u2, u3])

    assert table["uid"].nunique() == 2                     # only the 2 wide users
    assert set(table["uid"]) == {"cg-017", "cg-018"}
    assert list(table.columns) == META_COLS + UNIFIED_COLUMNS
    # NP-only column stays NaN across the CGMacros-only table
    assert table["temp"].isna().all()


def test_build_unified_table_raises_when_nothing_wide():
    u = LoadedUser("np-003", "nature_paper", "003")
    with pytest.raises(ValueError):
        build_unified_table([u])


# ── Real data (guarded) ───────────────────────────────────────────────────────

def test_real_cgmacros_user_loads_and_conforms():
    ids = discover_users("cgmacros")
    if not ids:
        pytest.skip("No CGMacros users on disk.")
    user = load_user("cgmacros", ids[0])
    assert user.uid == f"cg-{ids[0]}"
    assert user.is_wide
    assert "glucose_mg_dl" in user.frame.columns
    # conforms cleanly to the unified schema
    unified = unify_columns(user.frame)
    assert list(unified.columns) == META_COLS + UNIFIED_COLUMNS
