"""
Question 1 – Data Storytelling: Let's Talk Forum Engagement Since 2022
=======================================================================
Run AFTER scrape_letstalk.py has produced:
  - letstalk_topics.csv
  - letstalk_posts.csv

Outputs (all saved to ./outputs/):
  - monthly_topics.csv
  - monthly_posts.csv
  - user_cohort.csv
  - category_breakdown.csv
  - engagement_summary.csv
  - top_threads.csv

These CSVs feed the HTML data story (data_story_q1.html).
"""

import pandas as pd
import numpy as np
import re
from html.parser import HTMLParser
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("Loading data...")
topics = pd.read_csv("letstalk_topics.csv", parse_dates=["created_at", "last_posted_at"])
posts  = pd.read_csv("letstalk_posts.csv",  parse_dates=["created_at", "updated_at"])

# ── 2. CLEAN DATES ────────────────────────────────────────────────────────────
topics["year_month"] = topics["created_at"].dt.to_period("M")
posts["year_month"]  = posts["created_at"].dt.to_period("M")

# Filter to 2022 onwards
topics = topics[topics["created_at"].dt.year >= 2022].copy()
posts  = posts[posts["created_at"].dt.year >= 2022].copy()

# ── 3. STRIP HTML FROM POST CONTENT ──────────────────────────────────────────
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return " ".join(self.fed)

def strip_html(html):
    if pd.isna(html):
        return ""
    s = MLStripper()
    s.feed(str(html))
    return s.get_data().strip()

posts["text"] = posts["cooked"].apply(strip_html)
posts["word_count"] = posts["text"].apply(lambda x: len(x.split()))

# ── 4. MONTHLY ACTIVITY ───────────────────────────────────────────────────────
monthly_topics = (
    topics.groupby("year_month")
    .agg(new_topics=("topic_id", "count"),
         total_views=("views", "sum"),
         total_likes=("like_count", "sum"))
    .reset_index()
)

monthly_posts = (
    posts.groupby("year_month")
    .agg(new_posts=("post_id", "count"),
         unique_users=("username", "nunique"),
         avg_word_count=("word_count", "mean"))
    .reset_index()
)

monthly = monthly_topics.merge(monthly_posts, on="year_month", how="outer").fillna(0)
monthly["year_month"] = monthly["year_month"].astype(str)
monthly.to_csv("outputs/monthly_activity.csv", index=False)
print(f"  Monthly activity: {len(monthly)} months")

# ── 5. CATEGORY BREAKDOWN ─────────────────────────────────────────────────────
cat_stats = (
    topics.groupby("category_name")
    .agg(topics=("topic_id", "count"),
         total_views=("views", "sum"),
         total_replies=("reply_count", "sum"),
         total_likes=("like_count", "sum"))
    .reset_index()
    .sort_values("topics", ascending=False)
)
cat_stats.to_csv("outputs/category_breakdown.csv", index=False)

# ── 6. USER COHORT / RETENTION PROXY ─────────────────────────────────────────
# First post month = cohort; then check if user posted in subsequent months
posts["cohort"] = posts.groupby("username")["created_at"].transform("min").dt.to_period("M")
posts["cohort_str"] = posts["cohort"].astype(str)
posts["post_month_str"] = posts["year_month"].astype(str)

cohort_activity = (
    posts.groupby(["cohort_str", "post_month_str"])["username"]
    .nunique()
    .reset_index(name="active_users")
)

# Cohort size (users who joined that month)
cohort_sizes = posts.groupby("cohort_str")["username"].nunique().reset_index(name="cohort_size")
cohort_activity = cohort_activity.merge(cohort_sizes, on="cohort_str")
cohort_activity["retention_rate"] = cohort_activity["active_users"] / cohort_activity["cohort_size"]
cohort_activity.to_csv("outputs/user_cohort.csv", index=False)
print(f"  Cohort rows: {len(cohort_activity)}")

# ── 7. TOP THREADS ────────────────────────────────────────────────────────────
top_threads = (
    topics[["topic_id", "title", "category_name", "views", "reply_count",
            "like_count", "posts_count", "created_at", "url"]]
    .sort_values("views", ascending=False)
    .head(20)
)
top_threads["created_at"] = top_threads["created_at"].astype(str)
top_threads.to_csv("outputs/top_threads.csv", index=False)

# ── 8. ENGAGEMENT DEPTH RATIO ─────────────────────────────────────────────────
# Ratio of replies to views (higher = more engaged community)
topics["engagement_ratio"] = topics["reply_count"] / (topics["views"] + 1)
monthly_er = (
    topics.groupby("year_month")["engagement_ratio"]
    .mean()
    .reset_index(name="avg_engagement_ratio")
)
monthly_er["year_month"] = monthly_er["year_month"].astype(str)
monthly_er.to_csv("outputs/engagement_ratio.csv", index=False)

# ── 9. REPLY LATENCY ──────────────────────────────────────────────────────────
# How quickly does a topic get its first reply?
first_reply = (
    posts[posts["post_number"] == 2]
    .groupby("topic_id")["created_at"]
    .min()
    .reset_index(name="first_reply_at")
)
latency = topics[["topic_id", "created_at"]].merge(first_reply, on="topic_id", how="left")
latency["hours_to_reply"] = (
    latency["first_reply_at"] - latency["created_at"]
).dt.total_seconds() / 3600
latency["year_month"] = latency["created_at"].dt.to_period("M").astype(str)
monthly_latency = (
    latency.groupby("year_month")["hours_to_reply"]
    .median()
    .reset_index(name="median_hours_to_first_reply")
)
monthly_latency.to_csv("outputs/reply_latency.csv", index=False)

# ── 10. SUMMARY STATS ─────────────────────────────────────────────────────────
summary = {
    "total_topics":        len(topics),
    "total_posts":         len(posts),
    "total_unique_users":  posts["username"].nunique(),
    "total_views":         int(topics["views"].sum()),
    "total_likes":         int(topics["like_count"].sum()),
    "date_range_start":    str(topics["created_at"].min().date()),
    "date_range_end":      str(topics["created_at"].max().date()),
    "avg_replies_per_topic": round(topics["reply_count"].mean(), 2),
    "median_hours_to_reply": round(latency["hours_to_reply"].median(), 1),
}
pd.DataFrame([summary]).to_csv("outputs/engagement_summary.csv", index=False)

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

print("\n✅ All outputs saved to ./outputs/")
print("   Next: open data_story_q1.html in a browser")
