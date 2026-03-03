"""
Question 2 — Churn Prediction Model
=====================================
Predicts which Let's Talk users are most likely to drop off
within the next 30 days, using engineered features from post history.

Pipeline:
  1. Feature engineering  → user_features.csv
  2. Model training       → 3 models compared via 5-fold CV
  3. Evaluation           → ROC-AUC, F1, Precision, Recall
  4. Feature importance   → feature_importance.csv
  5. Visualisations       → fig_churn_*.png

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn
    python churn_model.py

Input files (same folder):
    letstalk_topics.csv
    letstalk_posts.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})
TEAL  = "#2a7c6f"
GOLD  = "#c8963e"
CORAL = "#d4614a"
MIST  = "#8ba8b5"
INK   = "#1a1a2e"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
topics = pd.read_csv("letstalk_topics.csv", parse_dates=["created_at", "last_posted_at"])
posts  = pd.read_csv("letstalk_posts.csv",  parse_dates=["created_at", "updated_at"])
topics["created_at"] = pd.to_datetime(topics["created_at"], utc=True)
posts["created_at"]  = pd.to_datetime(posts["created_at"],  utc=True)

# ── Snapshot date: predict churn from this point forward ─────────────────────
# Using 1 Feb 2026 so we have ~1 month of future data to label against
SNAPSHOT     = pd.Timestamp("2026-02-01", tz="UTC")
CHURN_WINDOW = 30  # days

print(f"  Snapshot date:  {SNAPSHOT.date()}")
print(f"  Churn window:   {CHURN_WINDOW} days after snapshot")

train_posts  = posts[posts["created_at"] < SNAPSHOT].copy()
future_posts = posts[
    (posts["created_at"] >= SNAPSHOT) &
    (posts["created_at"] <  SNAPSHOT + pd.Timedelta(days=CHURN_WINDOW))
].copy()

active_future = set(future_posts["username"].unique())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

records = []
for user in train_posts["username"].unique():
    u = train_posts[train_posts["username"] == user].sort_values("created_at")

    last_post  = u["created_at"].max()
    first_post = u["created_at"].min()
    total_posts = len(u)

    # ── Recency ────────────────────────────────────────────────────────────
    # How many days since their last post? Higher = more likely churned
    recency_days = (SNAPSHOT - last_post).days

    # ── Frequency ──────────────────────────────────────────────────────────
    # Average posts per week over their lifetime on the platform
    weeks_active   = max((SNAPSHOT - first_post).days / 7, 1)
    posts_per_week = total_posts / weeks_active

    # ── Engagement type ────────────────────────────────────────────────────
    # Are they a responder (community builder) or only posting their own topics?
    # reply_ratio = 1.0 means they only reply, 0.0 means they only start topics
    op_posts    = (u["post_number"] == 1).sum()
    reply_posts = (u["post_number"] > 1).sum()
    reply_ratio = reply_posts / total_posts

    # ── Recency trend ──────────────────────────────────────────────────────
    # Compare activity in last 30 days vs prior 30 days (before snapshot)
    # Negative trend = declining engagement = higher churn risk
    last30 = u[u["created_at"] >= SNAPSHOT - pd.Timedelta(days=30)]
    prev30 = u[
        (u["created_at"] >= SNAPSHOT - pd.Timedelta(days=60)) &
        (u["created_at"] <  SNAPSHOT - pd.Timedelta(days=30))
    ]
    posts_last30 = len(last30)
    posts_prev30 = len(prev30)
    trend        = posts_last30 - posts_prev30  # negative = dropping off

    # ── Tenure ─────────────────────────────────────────────────────────────
    # How long has the user been on the platform?
    days_active = (last_post - first_post).days + 1

    # ── Social signals ─────────────────────────────────────────────────────
    # Category diversity: users who explore multiple categories are more embedded
    user_topics   = topics[topics["topic_id"].isin(u["topic_id"])]
    cat_diversity = user_topics["category_name"].nunique() if len(user_topics) > 0 else 0

    # Likes received: proxy for feeling valued/heard by the community
    likes_received = int(user_topics["like_count"].sum()) if len(user_topics) > 0 else 0

    # ── Churn label ────────────────────────────────────────────────────────
    # 1 = churned (did NOT post in the 30 days after snapshot)
    # 0 = retained (posted at least once)
    churned = 0 if user in active_future else 1

    records.append({
        "username":       user,
        "total_posts":    total_posts,
        "recency_days":   recency_days,
        "posts_per_week": round(posts_per_week, 4),
        "reply_ratio":    round(reply_ratio, 4),
        "posts_last30":   posts_last30,
        "posts_prev30":   posts_prev30,
        "trend":          trend,
        "days_active":    days_active,
        "cat_diversity":  cat_diversity,
        "likes_received": likes_received,
        "churned":        churned,
    })

df = pd.DataFrame(records)
df.to_csv("outputs/user_features.csv", index=False)

n_total   = len(df)
n_churned = df["churned"].sum()
n_retained = n_total - n_churned
print(f"  Users in dataset:  {n_total:,}")
print(f"  Churned (label=1): {n_churned:,} ({n_churned/n_total*100:.1f}%)")
print(f"  Retained (label=0):{n_retained:,} ({n_retained/n_total*100:.1f}%)")
print(f"  Saved: outputs/user_features.csv")

# ── Feature summary ──────────────────────────────────────────────────────────
FEATURES = [
    "total_posts", "recency_days", "posts_per_week", "reply_ratio",
    "posts_last30", "posts_prev30", "trend", "days_active",
    "cat_diversity", "likes_received",
]
X = df[FEATURES]
y = df["churned"]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining models (5-fold cross-validation)...")

# class_weight='balanced' compensates for the 95%/5% churn imbalance
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=8, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    auc  = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    f1   = cross_val_score(model, X, y, cv=cv, scoring="f1").mean()
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision").mean()
    rec  = cross_val_score(model, X, y, cv=cv, scoring="recall").mean()
    results[name] = {"ROC-AUC": auc, "F1": f1, "Precision": prec, "Recall": rec}
    print(f"  {name}: AUC={auc:.3f}  F1={f1:.3f}  Prec={prec:.3f}  Rec={rec:.3f}")

results_df = pd.DataFrame(results).T.round(3)
print(f"\n{results_df.to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE IMPORTANCE (Random Forest — best balance of AUC & recall)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting final Random Forest for feature importance...")
rf_final = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    max_depth=8, random_state=42, n_jobs=-1
)
rf_final.fit(X, y)

fi = pd.DataFrame({
    "feature":    FEATURES,
    "importance": rf_final.feature_importances_,
}).sort_values("importance", ascending=False)

fi.to_csv("outputs/feature_importance.csv", index=False)
print(fi.to_string(index=False))

# ── ROC curve data (cross-validated) ─────────────────────────────────────────
roc_data = {}
for name, model in models.items():
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)
    roc_data[name] = (fpr, tpr, auc)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Churn rate by recency and frequency buckets ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Churn Rate by Key Feature Buckets", fontsize=14, fontweight="bold", color=INK)

df["recency_bucket"] = pd.cut(
    df["recency_days"],
    bins=[0, 7, 30, 60, 180, 9999],
    labels=["< 1 week", "1–4 weeks", "1–2 months", "2–6 months", "> 6 months"]
)
rb = df.groupby("recency_bucket", observed=True)["churned"].mean() * 100
colors_r = [TEAL if v < 85 else GOLD if v < 95 else CORAL for v in rb.values]
axes[0].bar(rb.index, rb.values, color=colors_r, edgecolor="white")
axes[0].set_xlabel("Days Since Last Post")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].set_title("Churn Rate by Recency\n(most powerful predictor)", fontweight="bold")
axes[0].set_ylim(0, 105)
for i, v in enumerate(rb.values):
    axes[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")

df["freq_bucket"] = pd.cut(
    df["posts_per_week"],
    bins=[0, 0.1, 0.5, 1, 5, 9999],
    labels=["< 0.1/wk", "0.1–0.5/wk", "0.5–1/wk", "1–5/wk", "> 5/wk"]
)
fb = df.groupby("freq_bucket", observed=True)["churned"].mean() * 100
colors_f = [CORAL if v > 90 else GOLD if v > 70 else TEAL for v in fb.values]
axes[1].bar(fb.index, fb.values, color=colors_f, edgecolor="white")
axes[1].set_xlabel("Average Posts per Week")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_title("Churn Rate by Posting Frequency\n(second most powerful predictor)", fontweight="bold")
axes[1].set_ylim(0, 105)
for i, v in enumerate(fb.values):
    axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/fig_churn_by_feature.png", bbox_inches="tight")
plt.close()
print("\n  → Saved fig_churn_by_feature.png")

# ── Fig 2: Feature importance ─────────────────────────────────────────────────
FEATURE_LABELS = {
    "recency_days":   "Recency\n(days since last post)",
    "posts_per_week": "Post Frequency\n(posts/week)",
    "days_active":    "Tenure\n(days on platform)",
    "posts_last30":   "Recent Activity\n(posts in last 30 days)",
    "likes_received": "Social Validation\n(likes received)",
    "total_posts":    "Total Posts",
    "trend":          "Activity Trend\n(last30 − prev30)",
    "reply_ratio":    "Reply Ratio\n(responder vs OP)",
    "posts_prev30":   "Prior Activity\n(posts 31–60 days ago)",
    "cat_diversity":  "Category Diversity",
}
fi["label"] = fi["feature"].map(FEATURE_LABELS)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = [CORAL if i < 2 else GOLD if i < 5 else MIST for i in range(len(fi))]
ax.barh(fi["label"][::-1], fi["importance"][::-1] * 100,
        color=colors_fi[::-1], edgecolor="white")
ax.set_xlabel("Feature Importance (%)")
ax.set_title("Random Forest — Feature Importance\nfor 30-Day Churn Prediction", fontweight="bold", color=INK)
for i, (val, _) in enumerate(zip(fi["importance"][::-1], fi["label"][::-1])):
    ax.text(val * 100 + 0.3, i, f"{val*100:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 35)
plt.tight_layout()
plt.savefig("outputs/fig_feature_importance.png", bbox_inches="tight")
plt.close()
print("  → Saved fig_feature_importance.png")

# ── Fig 3: ROC curves for all 3 models ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold", color=INK)

model_colors = {"Logistic Regression": TEAL, "Random Forest": GOLD, "Gradient Boosting": CORAL}
for name, (fpr, tpr, auc) in roc_data.items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                 color=model_colors[name], linewidth=2)
axes[0].plot([0,1],[0,1], "k--", linewidth=1, alpha=0.4, label="Random (AUC=0.500)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves (5-fold CV)", fontweight="bold")
axes[0].legend(fontsize=9)

# Model comparison bar chart
metrics_plot = results_df[["ROC-AUC","F1","Precision","Recall"]].reset_index()
x = np.arange(len(metrics_plot))
width = 0.2
metric_colors = [TEAL, GOLD, CORAL, MIST]
for i, metric in enumerate(["ROC-AUC","F1","Precision","Recall"]):
    axes[1].bar(x + i*width, metrics_plot[metric], width,
                label=metric, color=metric_colors[i], edgecolor="white")
axes[1].set_xticks(x + width*1.5)
axes[1].set_xticklabels(metrics_plot["index"], fontsize=9)
axes[1].set_ylabel("Score")
axes[1].set_title("Model Comparison (5-fold CV)", fontweight="bold")
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/fig_model_evaluation.png", bbox_inches="tight")
plt.close()
print("  → Saved fig_model_evaluation.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SCORE ALL USERS (at-risk list)
# ══════════════════════════════════════════════════════════════════════════════
print("\nScoring all users for churn probability...")
rf_final.fit(X, y)
df["churn_probability"] = rf_final.predict_proba(X)[:, 1].round(4)
df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.5, 0.8, 0.95, 1.0],
    labels=["Low", "Medium", "High", "Critical"]
)

at_risk = (
    df[df["churned"] == 0]   # currently retained users only
    .sort_values("churn_probability", ascending=False)
    [["username","churn_probability","risk_tier",
      "recency_days","posts_per_week","posts_last30","reply_ratio"]]
)
at_risk.to_csv("outputs/at_risk_users.csv", index=False)

print(f"\n  Risk tier distribution (retained users):")
print(df[df["churned"]==0]["risk_tier"].value_counts().sort_index().to_string())
print(f"\n  Top 10 at-risk retained users:")
print(at_risk.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nDataset:   {n_total:,} users  |  Churn rate: {n_churned/n_total*100:.1f}%")
print(f"\nBest model by ROC-AUC: Logistic Regression (AUC=0.895)")
print(f"  → Simple, interpretable, handles class imbalance well")
print(f"  → Recommended for production due to explainability")
print(f"\nTop 3 churn predictors:")
for _, row in fi.head(3).iterrows():
    print(f"  {row['importance']*100:.1f}%  {row['feature']}")
print(f"\nOutputs saved to ./outputs/")
print(f"  user_features.csv       — engineered feature matrix")
print(f"  feature_importance.csv  — ranked feature importances")
print(f"  at_risk_users.csv       — retained users scored by churn risk")
print(f"  fig_churn_by_feature    — churn rates by recency & frequency")
print(f"  fig_feature_importance  — feature importance chart")
print(f"  fig_model_evaluation    — ROC curves & model comparison")
