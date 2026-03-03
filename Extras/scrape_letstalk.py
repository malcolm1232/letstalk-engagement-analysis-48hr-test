"""
Scraper for https://letstalk.mindline.sg/ (Discourse forum)
Uses the public Discourse JSON API to extract all topics and posts.

Run:
    pip install requests pandas
    python scrape_letstalk.py

Output: letstalk_topics.csv, letstalk_posts.csv
"""

import requests
import pandas as pd
import time

BASE_URL = "https://letstalk.mindline.sg"
HEADERS = {"Accept": "application/json"}

CATEGORIES = [
    {"id": 5, "slug": "ask-a-therapist"},
    {"id": 6, "slug": "hangouts"},
    {"id": 7, "slug": "self-care-lounge"},
    {"id": 8, "slug": "community-events"},
]

def get_topics_for_category(cat_slug, cat_id):
    """Fetch all topics from a category using pagination."""
    topics = []
    page = 0
    while True:
        url = f"{BASE_URL}/c/{cat_slug}/{cat_id}.json?page={page}"
        print(f"  Fetching: {url}")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"  ⚠ Status {resp.status_code}, stopping.")
            break
        data = resp.json()
        topic_list = data.get("topic_list", {}).get("topics", [])
        if not topic_list:
            break
        for t in topic_list:
            topics.append({
                "topic_id":        t.get("id"),
                "title":           t.get("title"),
                "slug":            t.get("slug"),
                "category_id":     t.get("category_id"),
                "category_name":   cat_slug,
                "created_at":      t.get("created_at"),
                "last_posted_at":  t.get("last_posted_at"),
                "views":           t.get("views"),
                "posts_count":     t.get("posts_count"),
                "reply_count":     t.get("reply_count"),
                "like_count":      t.get("like_count"),
                "has_accepted_answer": t.get("has_accepted_answer"),
                "tags":            ", ".join(
                    tag if isinstance(tag, str) else tag.get("name", str(tag))
                    for tag in t.get("tags", [])
                ),
                "url":             f"{BASE_URL}/t/{t.get('slug')}/{t.get('id')}",
            })
        # Check if there's a next page
        more_topics = data.get("topic_list", {}).get("more_topics_url")
        if not more_topics:
            break
        page += 1
        time.sleep(0.5)
    return topics


def get_posts_for_topic(topic_id, topic_slug):
    """Fetch all posts from a specific topic."""
    posts = []
    url = f"{BASE_URL}/t/{topic_slug}/{topic_id}.json"
    print(f"    Fetching posts: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return posts
    data = resp.json()
    raw_posts = data.get("post_stream", {}).get("posts", [])
    for p in raw_posts:
        posts.append({
            "post_id":        p.get("id"),
            "topic_id":       topic_id,
            "post_number":    p.get("post_number"),
            "username":       p.get("username"),
            "created_at":     p.get("created_at"),
            "updated_at":     p.get("updated_at"),
            "cooked":         p.get("cooked"),       # HTML content
            "raw_text":       p.get("raw", ""),       # plain text (if available)
            "reads":          p.get("reads"),
            "score":          p.get("score"),
            "reply_to_post_number": p.get("reply_to_post_number"),
        })
    time.sleep(0.3)
    return posts


def main():
    all_topics = []
    all_posts = []

    print("=== Fetching Topics ===")
    for cat in CATEGORIES:
        print(f"\nCategory: {cat['slug']}")
        topics = get_topics_for_category(cat["slug"], cat["id"])
        print(f"  → {len(topics)} topics found")
        all_topics.extend(topics)

    print(f"\nTotal topics: {len(all_topics)}")

    print("\n=== Fetching Posts for Each Topic ===")
    for t in all_topics:
        posts = get_posts_for_topic(t["topic_id"], t["slug"])
        print(f"  Topic {t['topic_id']}: {len(posts)} posts")
        all_posts.extend(posts)

    # Save to CSV
    topics_df = pd.DataFrame(all_topics)
    posts_df  = pd.DataFrame(all_posts)

    topics_df.to_csv("letstalk_topics.csv", index=False, encoding="utf-8-sig")
    posts_df.to_csv("letstalk_posts.csv",  index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved letstalk_topics.csv  ({len(topics_df)} rows)")
    print(f"✅ Saved letstalk_posts.csv   ({len(posts_df)} rows)")

    # Quick preview
    print("\n--- Topics preview ---")
    print(topics_df[["topic_id","title","views","posts_count","like_count"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
