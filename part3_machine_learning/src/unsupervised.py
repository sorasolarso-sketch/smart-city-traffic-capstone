"""
Capstone Part 3, Task 2 — Unsupervised machine learning.

1. K-means clustering of traffic conditions (hour, weather severity,
   traffic volume, temperature), with k selected from the elbow and
   silhouette curves rather than assumed.
2. Association rule mining over discretised time of day, day type and
   weather, mining rules that predict congestion level.

Usage
-----
    python part3_machine_learning/src/unsupervised.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR.parent.parent / "part2_python"))

import common  # noqa: E402
from logging_config import configure_logging, log_stage_banner  # noqa: E402
import viz_style  # noqa: E402

logger = logging.getLogger(__name__)

LOG_FILE = common.PART3_DIR / "logs" / "part3.log"
CLUSTER_FEATURES = ["hour", "weather_severity", "traffic_volume", "temp"]
K_RANGE = range(2, 11)
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
def choose_k(X_scaled: np.ndarray) -> tuple[int, dict]:
    """Select k from the elbow and silhouette curves."""
    inertias, silhouettes, calinski = {}, {}, {}

    # Silhouette on the full 40,000 rows is slow and adds nothing over a
    # large sample, so it is computed on a fixed random subsample.
    rng = np.random.default_rng(common.RANDOM_STATE)
    sample_idx = rng.choice(len(X_scaled), size=min(6000, len(X_scaled)),
                            replace=False)

    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=common.RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        inertias[k] = float(model.inertia_)
        silhouettes[k] = float(silhouette_score(
            X_scaled[sample_idx], labels[sample_idx]))
        calinski[k] = float(calinski_harabasz_score(X_scaled, labels))
        logger.debug("k=%s — inertia %.1f, silhouette %.4f, Calinski-Harabasz %.1f",
                     k, inertias[k], silhouettes[k], calinski[k])

    best_silhouette_k = max(silhouettes, key=silhouettes.get)

    # Elbow via the largest perpendicular distance from the line joining the
    # first and last points of the inertia curve — the "kneedle" construction.
    ks = np.array(list(inertias))
    values = np.array([inertias[k] for k in ks])
    norm_k = (ks - ks.min()) / (ks.max() - ks.min())
    norm_v = (values - values.min()) / (values.max() - values.min())
    line = np.vstack([norm_k[-1] - norm_k[0], norm_v[-1] - norm_v[0]])
    line = line / np.linalg.norm(line)
    points = np.vstack([norm_k - norm_k[0], norm_v - norm_v[0]])
    projection = (points * line).sum(axis=0) * line
    distances = np.linalg.norm(points - projection, axis=0)
    elbow_k = int(ks[int(np.argmax(distances))])

    logger.info("Elbow suggests k=%s; silhouette peaks at k=%s (%.4f)",
                elbow_k, best_silhouette_k, silhouettes[best_silhouette_k])

    # An honest note on cluster quality. Silhouette runs from -1 to 1;
    # values below about 0.5 mean the groups overlap substantially rather
    # than forming crisp, well-separated clusters.
    if silhouettes[best_silhouette_k] < 0.5:
        logger.warning(
            "Best silhouette across all k is only %.3f. Traffic conditions form "
            "a continuum rather than well-separated natural groups, so these "
            "clusters should be read as a useful partition of a gradient, not "
            "as discovered categories.", silhouettes[best_silhouette_k],
        )

    chosen = elbow_k
    if silhouettes[best_silhouette_k] - silhouettes[elbow_k] > 0.05:
        logger.warning(
            "Silhouette at k=%s (%.4f) is materially better than at the elbow "
            "k=%s (%.4f); choosing the silhouette optimum",
            best_silhouette_k, silhouettes[best_silhouette_k],
            elbow_k, silhouettes[elbow_k],
        )
        chosen = best_silhouette_k

    diagnostics = {
        "inertia": inertias,
        "silhouette": silhouettes,
        "calinski_harabasz": calinski,
        "elbow_k": elbow_k,
        "silhouette_best_k": int(best_silhouette_k),
        "chosen_k": int(chosen),
    }
    return int(chosen), diagnostics


def describe_clusters(df: pd.DataFrame, k: int) -> dict:
    """Produce a plain-language reading of each cluster."""
    profiles = {}
    for cluster_id in range(k):
        subset = df[df["cluster"] == cluster_id]
        share = 100 * len(subset) / len(df)
        peak_hours = subset["hour"].value_counts().nlargest(4).index.tolist()
        weather_mix = subset["weather_main"].value_counts(normalize=True).nlargest(3)

        mean_volume = subset["traffic_volume"].mean()
        mean_hour = subset["hour"].mean()
        mean_severity = subset["weather_severity"].mean()
        mean_temp_c = subset["temp"].mean() - 273.15
        weekend_share = 100 * subset["is_weekend"].mean()

        # Name the cluster from its own centre rather than by hand, so the
        # labels stay correct if the data or k changes.
        if mean_volume < 1200:
            volume_word = "near-empty"
        elif mean_volume < 3000:
            volume_word = "light"
        elif mean_volume < 4800:
            volume_word = "moderate"
        else:
            volume_word = "heavy"

        if mean_severity < 1.5:
            weather_word = "fair weather"
        elif mean_severity < 4:
            weather_word = "mixed weather"
        else:
            weather_word = "adverse weather"

        if 5 <= mean_hour < 11:
            time_word = "morning"
        elif 11 <= mean_hour < 15:
            time_word = "midday"
        elif 15 <= mean_hour < 20:
            time_word = "afternoon and evening"
        else:
            time_word = "overnight"

        # Temperature is what separates the two fair-weather daytime
        # clusters, so it has to appear in the label or they read as
        # duplicates of one another.
        if mean_temp_c < 0:
            temp_word = "sub-zero"
        elif mean_temp_c < 10:
            temp_word = "cool"
        elif mean_temp_c < 18:
            temp_word = "mild"
        else:
            temp_word = "warm"

        profiles[str(cluster_id)] = {
            "hours": int(len(subset)),
            "share_pct": float(share),
            "mean_traffic_volume": float(mean_volume),
            "mean_hour": float(mean_hour),
            "modal_hours": [int(h) for h in peak_hours],
            "mean_weather_severity": float(mean_severity),
            "mean_temp_celsius": float(mean_temp_c),
            "weekend_share_pct": float(weekend_share),
            "top_weather": {str(k_): round(float(v), 3) for k_, v in weather_mix.items()},
            "congestion_mix": {
                str(k_): round(float(v), 3) for k_, v in
                subset["congestion_category"].value_counts(normalize=True).items()
            },
            "label": (f"{time_word.title()} {volume_word} traffic, "
                      f"{temp_word} {weather_word}"),
        }
        logger.info(
            "Cluster %s — %s hours (%.1f%%): %s | mean volume %.0f, mean hour "
            "%.1f, severity %.2f, %.0f%% weekend",
            cluster_id, f"{len(subset):,}", share, profiles[str(cluster_id)]["label"],
            mean_volume, mean_hour, mean_severity, weekend_share,
        )
    return profiles


def plot_k_selection(diagnostics: dict) -> None:
    ks = list(diagnostics["inertia"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    axes[0].plot(ks, [diagnostics["inertia"][k] for k in ks],
                 color=viz_style.SERIES[0], marker="o")
    axes[0].axvline(diagnostics["chosen_k"], color=viz_style.SERIES[1],
                    linestyle="--", linewidth=1.5)
    axes[0].set_title("Inertia (elbow)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Within-cluster sum of squares")

    axes[1].plot(ks, [diagnostics["silhouette"][k] for k in ks],
                 color=viz_style.SERIES[0], marker="o")
    axes[1].axvline(diagnostics["chosen_k"], color=viz_style.SERIES[1],
                    linestyle="--", linewidth=1.5,
                    label=f"chosen k = {diagnostics['chosen_k']}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Mean silhouette")
    axes[1].legend()

    fig.suptitle("Choosing the number of traffic-condition clusters",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", y=1.04)
    path = common.FIGURE_DIR / "10_kmeans_selection.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_clusters(df: pd.DataFrame, profiles: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id].sample(
            n=min(1800, int((df["cluster"] == cluster_id).sum())),
            random_state=common.RANDOM_STATE,
        )
        jitter = np.random.default_rng(cluster_id).normal(0, 0.16, len(subset))
        ax.scatter(subset["hour"] + jitter, subset["traffic_volume"],
                   s=7, alpha=0.32, linewidths=0,
                   color=viz_style.SERIES[cluster_id % len(viz_style.SERIES)],
                   label=f"{cluster_id}: {profiles[str(cluster_id)]['label']}")

    ax.set_title("K-means separates the corridor into distinct operating states")
    viz_style.add_subtitle(ax, "Each point is one hour, positioned by clock time and volume, coloured by cluster")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="both")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8.5,
              ncol=2, markerscale=2.2, frameon=False)
    path = common.FIGURE_DIR / "11_kmeans_clusters.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def run_kmeans(df: pd.DataFrame) -> dict:
    log_stage_banner(logger, "Task 2a — K-means clustering of traffic conditions")

    X = df[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Clustering on %s features: %s", len(CLUSTER_FEATURES), CLUSTER_FEATURES)
    logger.debug("Feature means before scaling: %s",
                 dict(zip(CLUSTER_FEATURES, X.mean(axis=0).round(3))))

    k, diagnostics = choose_k(X_scaled)
    model = KMeans(n_clusters=k, n_init=20, random_state=common.RANDOM_STATE)
    df["cluster"] = model.fit_predict(X_scaled)
    logger.info("Final K-means fitted with k=%s, inertia %.1f", k, model.inertia_)

    centres = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_), columns=CLUSTER_FEATURES
    )
    for i, row in centres.iterrows():
        logger.debug("Cluster %s centre (original units): %s", i, row.round(2).to_dict())

    profiles = describe_clusters(df, k)
    plot_k_selection(diagnostics)
    plot_clusters(df, profiles)

    return {
        "features": CLUSTER_FEATURES,
        "k_selection": diagnostics,
        "chosen_k": k,
        "inertia": float(model.inertia_),
        "silhouette_at_chosen_k": diagnostics["silhouette"][k],
        "cluster_centres_original_units": centres.round(3).to_dict("index"),
        "cluster_profiles": profiles,
    }


# ---------------------------------------------------------------------------
# Association rules
# ---------------------------------------------------------------------------
def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise the data into one basket of items per hour."""
    items = pd.DataFrame(index=df.index)

    for band in df["part_of_day"].unique():
        items[f"time={band}"] = (df["part_of_day"] == band)

    items["day=Weekend"] = df["is_weekend"] == 1
    items["day=Weekday"] = df["is_weekend"] == 0

    for condition in df["weather_main"].unique():
        count = int((df["weather_main"] == condition).sum())
        if count >= 200:
            items[f"weather={condition}"] = (df["weather_main"] == condition)
        else:
            logger.debug("Weather category '%s' excluded from rule mining "
                         "(only %s hours)", condition, count)

    items["weather=Precipitating"] = df["is_precipitating"] == 1
    items["weather=LowVisibility"] = df["is_low_visibility"] == 1
    items["temp=Freezing"] = df["is_freezing"] == 1
    items["holiday=Yes"] = df["is_holiday"] == 1

    for level in ["Low", "Medium", "High", "Severe"]:
        items[f"congestion={level}"] = (df["congestion_category"] == level)

    logger.info("Transaction matrix built — %s hours x %s items",
                f"{len(items):,}", items.shape[1])
    logger.debug("Items: %s", list(items.columns))
    return items.astype(bool)


def run_association_rules(df: pd.DataFrame) -> dict:
    log_stage_banner(logger, "Task 2b — Association rule mining")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        logger.error("mlxtend is not installed; run pip install mlxtend",
                     exc_info=True)
        raise

    transactions = build_transactions(df)
    frequent = apriori(transactions, min_support=MIN_SUPPORT, use_colnames=True,
                       max_len=4)
    logger.info("Apriori found %s frequent itemset(s) at min_support=%.2f",
                f"{len(frequent):,}", MIN_SUPPORT)

    if frequent.empty:
        logger.warning("No frequent itemsets at min_support=%.2f", MIN_SUPPORT)
        return {"rules": [], "min_support": MIN_SUPPORT}

    rules = association_rules(frequent, metric="confidence",
                              min_threshold=MIN_CONFIDENCE)
    logger.info("Generated %s rule(s) at min_confidence=%.2f",
                f"{len(rules):,}", MIN_CONFIDENCE)

    # Keep only rules whose conclusion is a congestion level, which is the
    # question the brief asks the mining to answer.
    congestion_items = {f"congestion={lvl}" for lvl in
                        ["Low", "Medium", "High", "Severe"]}
    rules["consequent_str"] = rules["consequents"].apply(
        lambda s: ", ".join(sorted(s)))
    rules["antecedent_str"] = rules["antecedents"].apply(
        lambda s: " AND ".join(sorted(s)))
    congestion_rules = rules[
        rules["consequents"].apply(lambda s: set(s).issubset(congestion_items))
        & ~rules["antecedents"].apply(lambda s: bool(set(s) & congestion_items))
    ].copy()

    logger.info("%s rule(s) conclude with a congestion level",
                f"{len(congestion_rules):,}")

    if congestion_rules.empty:
        logger.warning("No congestion rules survived filtering")
        return {"rules": [], "min_support": MIN_SUPPORT}

    top = congestion_rules.nlargest(15, "lift")
    logger.info("Top 5 rules by lift:")
    for _, row in top.head(5).iterrows():
        logger.info(
            "  %s -> %s | support %.3f, confidence %.3f, lift %.2f",
            row["antecedent_str"], row["consequent_str"],
            row["support"], row["confidence"], row["lift"],
        )

    plot_rules(top)

    records = []
    for _, row in top.iterrows():
        records.append({
            "antecedent": row["antecedent_str"],
            "consequent": row["consequent_str"],
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "hours_matching": int(round(row["support"] * len(df))),
            "plain_language": explain_rule(row, len(df)),
        })

    return {
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "frequent_itemsets": int(len(frequent)),
        "total_rules": int(len(rules)),
        "congestion_rules": int(len(congestion_rules)),
        "top_rules_by_lift": records,
    }


def explain_rule(row, total_rows: int) -> str:
    """Translate one rule into a sentence a non-specialist can act on."""
    antecedent = row["antecedent_str"].replace("time=", "").replace("day=", "")
    antecedent = antecedent.replace("weather=", "").replace("temp=", "")
    antecedent = antecedent.replace("holiday=Yes", "a public holiday")
    antecedent = antecedent.replace(" AND ", " and ")
    consequent = row["consequent_str"].replace("congestion=", "")
    hours = int(round(row["support"] * total_rows))
    return (
        f"When {antecedent.lower()}, congestion is {consequent.lower()} "
        f"{row['confidence']:.0%} of the time — {row['lift']:.2f} times more "
        f"often than the {consequent.lower()} level occurs overall. "
        f"This pattern covers {hours:,} hours."
    )


def plot_rules(top: pd.DataFrame) -> None:
    data = top.nlargest(10, "lift").sort_values("lift")
    labels = [
        (a[:44] + "…" if len(a) > 44 else a) + "  →  " + c.replace("congestion=", "")
        for a, c in zip(data["antecedent_str"], data["consequent_str"])
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.barh(range(len(data)), data["lift"], color=viz_style.SERIES[0], height=0.68)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(1.0, color=viz_style.TEXT_MUTED, linestyle="--", linewidth=1.2)
    ax.text(1.02, -0.8, "lift = 1 (no association)", fontsize=8.5,
            color=viz_style.TEXT_MUTED)

    for i, (lift, conf) in enumerate(zip(data["lift"], data["confidence"])):
        ax.text(lift + 0.02, i, f"{lift:.2f}  (conf {conf:.0%})",
                va="center", fontsize=8.5, color=viz_style.TEXT_SECONDARY)

    ax.set_title("Strongest association rules predicting congestion level")
    viz_style.add_subtitle(ax, "Ranked by lift; all rules meet 2% support and 30% confidence")
    ax.set_xlabel("Lift")
    ax.set_xlim(0, data["lift"].max() * 1.30)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    path = common.FIGURE_DIR / "12_association_rules.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Part 3 unsupervised analyses.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(log_file=LOG_FILE, debug=args.debug)
    viz_style.apply_style()
    common.ensure_directories()

    logger.info("#" * 72)
    logger.info("Capstone Part 3, Task 2 — unsupervised learning")
    logger.info("#" * 72)

    try:
        df = common.load_features()
        df = common.add_proxy_risk_label(df)
        kmeans_results = run_kmeans(df)
        rules_results = run_association_rules(df)

        results = {"kmeans": kmeans_results, "association_rules": rules_results}
        out_path = common.OUTPUT_DIR / "unsupervised_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", out_path)

        clustered = df[["date_time", "hour", "traffic_volume", "weather_main",
                        "congestion_category", "cluster"]]
        clustered.to_csv(common.OUTPUT_DIR / "clustered_hours.csv", index=False)
        logger.info("Cluster assignments written to %s",
                    common.OUTPUT_DIR / "clustered_hours.csv")

    except (FileNotFoundError, PermissionError, OSError):
        logger.error("Unsupervised stage aborted — file access problem", exc_info=True)
        return 3
    except (ValueError, KeyError, TypeError):
        logger.error("Unsupervised stage aborted — data problem", exc_info=True)
        return 5
    except ImportError:
        logger.error("Unsupervised stage aborted — missing library", exc_info=True)
        return 6

    logger.info("Unsupervised stage finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
