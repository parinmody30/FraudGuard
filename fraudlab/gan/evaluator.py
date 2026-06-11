"""Evaluate CTGAN realism: statistical fidelity + ML utility (TSTR / TRTS).

Two complementary views:

1. **Statistical similarity** between the real (simulator) soft-row distribution and the
   CTGAN-generated distribution:
   * KS statistic per continuous feature (lower = closer).
   * Jensen-Shannon divergence per categorical/boolean feature (lower = closer).
   * Correlation-matrix preservation over continuous features.
   * Mutual-information preservation: MI(feature; is_fraud) real vs synthetic.
   * Fraud-prevalence preservation and fraud-type-distribution preservation.

2. **ML utility** - train a detector on one source and test on the other:
   * TSTR: train on simulator, test on CTGAN.
   * TRTS: train on CTGAN, test on simulator.
   Reported with precision / recall / F1 / ROC-AUC / PR-AUC for Logistic Regression and
   Gradient Boosting. A real-on-real holdout is included as a reference ceiling.

Small TSTR/TRTS degradation versus the real-on-real reference indicates the synthetic data
carries the same per-row signal - i.e. realism transferred successfully.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.gan.metadata import GANMetadata

# Feature columns excluded from the ML-utility matrix (label + its proxy).
_LABEL = "is_fraud"
_LABEL_PROXY = "fraud_type"


# ---------------------------------------------------------------------------
# Statistical similarity
# ---------------------------------------------------------------------------

def _ks_continuous(real: pd.Series, synth: pd.Series) -> float:
    from scipy.stats import ks_2samp

    a = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(synth, errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def _js_categorical(real: pd.Series, synth: pd.Series) -> float:
    """Jensen-Shannon *divergence* (base 2, in [0, 1]) between category distributions."""
    from scipy.spatial.distance import jensenshannon

    cats = sorted(set(real.astype(str)) | set(synth.astype(str)))
    rp = real.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    sp = synth.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
    dist = jensenshannon(rp, sp, base=2)  # distance = sqrt(JS divergence)
    if np.isnan(dist):
        return 0.0
    return float(dist**2)


def _encode_for_corr(df: pd.DataFrame, metadata: GANMetadata) -> pd.DataFrame:
    cols = [c for c in metadata.continuous if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _correlation_distance(
    real: pd.DataFrame, synth: pd.DataFrame, metadata: GANMetadata
) -> dict[str, float]:
    rc = _encode_for_corr(real, metadata).corr().fillna(0.0).to_numpy()
    sc = _encode_for_corr(synth, metadata).corr().fillna(0.0).to_numpy()
    diff = np.abs(rc - sc)
    return {
        "mean_abs_corr_diff": round(float(diff.mean()), 4),
        "max_abs_corr_diff": round(float(diff.max()), 4),
        "frobenius": round(float(np.linalg.norm(rc - sc)), 4),
    }


def _build_categories(
    real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]
) -> dict[str, list[str]]:
    return {
        c: sorted(set(real[c].astype(str)) | set(synth[c].astype(str)))
        for c in cols
    }


def _encode_features(
    df: pd.DataFrame, metadata: GANMetadata, categories: dict[str, list[str]]
) -> pd.DataFrame:
    """Encode a frame into a numeric feature matrix aligned across real/synth.

    Excludes the label and its proxy. Categoricals/booleans are ordinal-coded against the
    shared ``categories``; datetime contributes epoch-seconds and hour-of-day.
    """
    out = pd.DataFrame(index=df.index)
    for c in metadata.columns:
        if c in (_LABEL, _LABEL_PROXY):
            continue
        if c in metadata.datetime:
            ts = pd.to_datetime(df[c], errors="coerce")
            out[c + "_epoch"] = (ts.astype("int64") / 1e9).astype("float64")
            out[c + "_hour"] = ts.dt.hour.fillna(0).astype("int64")
        elif c in metadata.categorical or c in metadata.boolean:
            mapping = {v: i for i, v in enumerate(categories[c])}
            out[c] = df[c].astype(str).map(mapping).fillna(-1).astype("int64")
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out


def _mutual_info_preservation(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata: GANMetadata,
    categories: dict[str, list[str]],
) -> dict[str, Any]:
    from sklearn.feature_selection import mutual_info_classif

    Xr = _encode_features(real, metadata, categories)
    Xs = _encode_features(synth, metadata, categories)
    yr = real[_LABEL].astype(int).to_numpy()
    ys = synth[_LABEL].astype(int).to_numpy()

    def _mi(X, y):
        if len(np.unique(y)) < 2:
            return np.zeros(X.shape[1])
        return mutual_info_classif(X.to_numpy(), y, random_state=0)

    mi_r = _mi(Xr, yr)
    mi_s = _mi(Xs, ys)
    diff = np.abs(mi_r - mi_s)
    per_feature = {
        col: {"real": round(float(r), 4), "synth": round(float(s), 4)}
        for col, r, s in zip(Xr.columns, mi_r, mi_s)
    }
    return {
        "mean_abs_mi_diff": round(float(diff.mean()), 4),
        "max_abs_mi_diff": round(float(diff.max()), 4),
        "per_feature": per_feature,
    }


def statistical_similarity(
    real: pd.DataFrame, synth: pd.DataFrame, metadata: GANMetadata
) -> dict[str, Any]:
    """Compute all statistical-fidelity metrics between real and synthetic frames."""
    ks = {
        c: round(_ks_continuous(real[c], synth[c]), 4)
        for c in metadata.continuous
        if c in real.columns and c in synth.columns
    }
    cat_cols = [
        c
        for c in (*metadata.categorical, *metadata.boolean)
        if c in real.columns and c in synth.columns
    ]
    js = {c: round(_js_categorical(real[c], synth[c]), 4) for c in cat_cols}

    corr = _correlation_distance(real, synth, metadata)
    categories = _build_categories(real, synth, cat_cols)
    mi = _mutual_info_preservation(real, synth, metadata, categories)

    real_rate = float(real[_LABEL].mean())
    synth_rate = float(synth[_LABEL].mean())

    ftypes = sorted(
        set(real[_LABEL_PROXY].dropna().astype(str)) | set(synth[_LABEL_PROXY].dropna().astype(str))
    )
    rd = real[_LABEL_PROXY].dropna().astype(str).value_counts(normalize=True)
    sd = synth[_LABEL_PROXY].dropna().astype(str).value_counts(normalize=True)
    fraud_type_dist = {
        t: {"real": round(float(rd.get(t, 0.0)), 4), "synth": round(float(sd.get(t, 0.0)), 4)}
        for t in ftypes
    }
    ftype_js = _js_categorical(
        real[_LABEL_PROXY].fillna("legit").astype(str),
        synth[_LABEL_PROXY].fillna("legit").astype(str),
    )

    return {
        "ks_continuous": ks,
        "mean_ks": round(float(np.nanmean(list(ks.values()))), 4) if ks else float("nan"),
        "js_categorical": js,
        "mean_js": round(float(np.mean(list(js.values()))), 4) if js else float("nan"),
        "correlation_preservation": corr,
        "mutual_information_preservation": mi,
        "fraud_prevalence": {
            "real": round(real_rate, 4),
            "synth": round(synth_rate, 4),
            "abs_diff": round(abs(real_rate - synth_rate), 4),
        },
        "fraud_type_preservation": {
            "distribution": fraud_type_dist,
            "js_divergence": round(ftype_js, 4),
        },
    }


# ---------------------------------------------------------------------------
# ML utility (TSTR / TRTS)
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (proba >= 0.5).astype(int)
    both = len(np.unique(y_true)) == 2
    return {
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if both else float("nan"),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if both else float("nan"),
    }


def _fit_predict(estimator, X_tr, y_tr, X_te) -> np.ndarray:
    estimator.fit(X_tr, y_tr)
    proba = estimator.predict_proba(X_te)
    idx = list(estimator.classes_).index(1) if 1 in estimator.classes_ else 1
    return proba[:, idx]


def _build_estimators(random_state: int = 42) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def ml_utility(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata: GANMetadata,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run TSTR, TRTS, and a real-on-real reference for each estimator."""
    from sklearn.model_selection import train_test_split

    cat_cols = [c for c in (*metadata.categorical, *metadata.boolean) if c in real.columns]
    categories = _build_categories(real, synth, cat_cols)

    Xr = _encode_features(real, metadata, categories)
    Xs = _encode_features(synth, metadata, categories)
    yr = real[_LABEL].astype(int).to_numpy()
    ys = synth[_LABEL].astype(int).to_numpy()

    # Real-on-real reference uses a stratified holdout of the real data.
    stratify = yr if len(np.unique(yr)) == 2 else None
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.3, random_state=random_state, stratify=stratify
    )

    results: dict[str, Any] = {}
    for name in ("logistic", "gradient_boosting"):
        ref = _build_estimators(random_state)[name]
        tstr = _build_estimators(random_state)[name]
        trts = _build_estimators(random_state)[name]

        ref_proba = _fit_predict(ref, Xr_tr, yr_tr, Xr_te)
        tstr_proba = _fit_predict(tstr, Xr, yr, Xs)   # train real, test synth
        trts_proba = _fit_predict(trts, Xs, ys, Xr)   # train synth, test real

        results[name] = {
            "reference_real_on_real": _metrics(yr_te, ref_proba),
            "TSTR": _metrics(ys, tstr_proba),
            "TRTS": _metrics(yr, trts_proba),
        }
    return results


# ---------------------------------------------------------------------------
# Orchestration + verdict
# ---------------------------------------------------------------------------

# Thresholds for a "ready for hybrid generation" verdict.
THRESHOLDS = {
    "mean_ks": 0.20,            # avg continuous KS
    "mean_js": 0.10,            # avg categorical JS divergence
    "fraud_prevalence_abs": 0.03,
    "fraud_type_js": 0.10,
    "tstr_roc_auc_drop": 0.15,  # max allowed drop vs real-on-real reference
}


def evaluate_ctgan(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    metadata: GANMetadata,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Full CTGAN evaluation: similarity + ML utility + a tuning recommendation."""
    similarity = statistical_similarity(real, synth, metadata)
    utility = ml_utility(real, synth, metadata, random_state=random_state)

    recommendation = _recommend(similarity, utility)
    return {
        "n_real": int(len(real)),
        "n_synth": int(len(synth)),
        "similarity": similarity,
        "ml_utility": utility,
        "thresholds": THRESHOLDS,
        "recommendation": recommendation,
    }


def _recommend(similarity: dict, utility: dict) -> dict[str, Any]:
    reasons: list[str] = []
    checks: dict[str, bool] = {}

    mean_ks = similarity["mean_ks"]
    checks["mean_ks_ok"] = not np.isnan(mean_ks) and mean_ks <= THRESHOLDS["mean_ks"]
    if not checks["mean_ks_ok"]:
        reasons.append(f"mean continuous KS {mean_ks} > {THRESHOLDS['mean_ks']}")

    mean_js = similarity["mean_js"]
    checks["mean_js_ok"] = not np.isnan(mean_js) and mean_js <= THRESHOLDS["mean_js"]
    if not checks["mean_js_ok"]:
        reasons.append(f"mean categorical JS {mean_js} > {THRESHOLDS['mean_js']}")

    prev = similarity["fraud_prevalence"]["abs_diff"]
    checks["prevalence_ok"] = prev <= THRESHOLDS["fraud_prevalence_abs"]
    if not checks["prevalence_ok"]:
        reasons.append(f"fraud-prevalence abs diff {prev} > {THRESHOLDS['fraud_prevalence_abs']}")

    ftjs = similarity["fraud_type_preservation"]["js_divergence"]
    checks["fraud_type_ok"] = ftjs <= THRESHOLDS["fraud_type_js"]
    if not checks["fraud_type_ok"]:
        reasons.append(f"fraud-type JS {ftjs} > {THRESHOLDS['fraud_type_js']}")

    # TSTR ROC-AUC should not collapse vs the real-on-real reference (use GBM).
    gbm = utility["gradient_boosting"]
    ref_auc = gbm["reference_real_on_real"]["roc_auc"]
    tstr_auc = gbm["TSTR"]["roc_auc"]
    drop = (
        round(ref_auc - tstr_auc, 4)
        if not (np.isnan(ref_auc) or np.isnan(tstr_auc))
        else float("nan")
    )
    checks["tstr_utility_ok"] = not np.isnan(drop) and drop <= THRESHOLDS["tstr_roc_auc_drop"]
    if not checks["tstr_utility_ok"]:
        reasons.append(
            f"TSTR ROC-AUC drop {drop} > {THRESHOLDS['tstr_roc_auc_drop']} (ref {ref_auc}, tstr {tstr_auc})"
        )

    ready = all(checks.values())
    return {
        "verdict": "READY FOR HYBRID GENERATION" if ready else "CTGAN NEEDS TUNING",
        "ready": ready,
        "checks": checks,
        "tstr_roc_auc_drop": drop,
        "failing_reasons": reasons,
    }


def format_markdown(report: dict) -> str:
    """Render an evaluation report dict as a Markdown document."""
    sim = report["similarity"]
    ml = report["ml_utility"]
    rec = report["recommendation"]
    lines: list[str] = []
    a = lines.append

    a("# CTGAN Realism Evaluation\n")
    a(f"- Real (simulator soft) rows: **{report['n_real']}**")
    a(f"- Synthetic (CTGAN) rows: **{report['n_synth']}**")
    a(f"- **Verdict: {rec['verdict']}**\n")

    a("## Statistical similarity\n")
    a(f"- Mean continuous KS: **{sim['mean_ks']}** (lower is better)")
    a(f"- Mean categorical JS divergence: **{sim['mean_js']}**")
    a(f"- Correlation preservation (mean abs diff): **{sim['correlation_preservation']['mean_abs_corr_diff']}**")
    a(f"- Mutual-info preservation (mean abs diff): **{sim['mutual_information_preservation']['mean_abs_mi_diff']}**")
    fp = sim["fraud_prevalence"]
    a(f"- Fraud prevalence real/synth: **{fp['real']} / {fp['synth']}** (abs diff {fp['abs_diff']})")
    a(f"- Fraud-type distribution JS: **{sim['fraud_type_preservation']['js_divergence']}**\n")

    a("### Per-continuous-feature KS\n")
    a("| feature | KS |")
    a("|---|---|")
    for f, v in sim["ks_continuous"].items():
        a(f"| {f} | {v} |")
    a("\n### Per-categorical-feature JS divergence\n")
    a("| feature | JS |")
    a("|---|---|")
    for f, v in sim["js_categorical"].items():
        a(f"| {f} | {v} |")

    a("\n## ML utility (TSTR / TRTS)\n")
    for model_name, res in ml.items():
        a(f"### {model_name}\n")
        a("| scenario | precision | recall | f1 | roc_auc | pr_auc |")
        a("|---|---|---|---|---|---|")
        for scen in ("reference_real_on_real", "TSTR", "TRTS"):
            m = res[scen]
            a(f"| {scen} | {m['precision']} | {m['recall']} | {m['f1']} | {m['roc_auc']} | {m['pr_auc']} |")
        a("")

    a("## Recommendation\n")
    a(f"**{rec['verdict']}**\n")
    a(f"- TSTR ROC-AUC drop vs reference (GBM): {rec['tstr_roc_auc_drop']}")
    a("- Checks:")
    for k, v in rec["checks"].items():
        a(f"  - {k}: {'PASS' if v else 'FAIL'}")
    if rec["failing_reasons"]:
        a("- Failing reasons:")
        for r in rec["failing_reasons"]:
            a(f"  - {r}")
    return "\n".join(lines)


def similarity_table(similarity: dict) -> pd.DataFrame:
    """Flatten per-feature similarity metrics into a tidy table for CSV export."""
    rows: list[dict[str, Any]] = []
    for feat, val in similarity["ks_continuous"].items():
        rows.append({"feature": feat, "kind": "continuous", "metric": "ks_statistic", "value": val})
    for feat, val in similarity["js_categorical"].items():
        rows.append({"feature": feat, "kind": "categorical", "metric": "js_divergence", "value": val})
    mi = similarity["mutual_information_preservation"]["per_feature"]
    for feat, vals in mi.items():
        rows.append(
            {
                "feature": feat,
                "kind": "mutual_info",
                "metric": "mi_abs_diff",
                "value": round(abs(vals["real"] - vals["synth"]), 4),
            }
        )
    return pd.DataFrame(rows, columns=["feature", "kind", "metric", "value"])
