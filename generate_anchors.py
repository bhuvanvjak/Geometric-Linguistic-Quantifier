"""
Fixed Anchor Generator for Geometric Linguistic Quantifier
===========================================================
KEY FIX over the old Anchor_generator.py:

OLD APPROACH (what crashed):
  - Encoded 562k sentences for Hindi
  - Wrote ALL sorted sentences into 4 huge .txt files on disk
  - Crashed with OSError: No space left on device

NEW APPROACH:
  - Samples a manageable 25k sentences per language  
  - Runs KMeans(n_clusters=2) on the embeddings
  - Saves ONLY the 2 cluster centroids (tiny 384-dim vectors) as .pt files
  - Zero text files written to disk
  - _baseline.pt  = centroid of cluster 0 (e.g. formal/news context)
  - _baseline1.pt = centroid of cluster 1 (e.g. casual/conversational context)
  => These are NOW genuinely different vectors, fixing the 0.9999 similarity bug
"""

import os
import json
import random
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

# ─── SETTINGS ────────────────────────────────────────────────────────────────
JSON_FILE      = "../../Data.json"       # path relative to this script
OUTPUT_DIR     = "."                     # save .pt files right here in Baselines/
MODEL_NAME     = "paraphrase-multilingual-MiniLM-L12-v2"
SAMPLE_SIZE    = 25000   # lines per language — big enough for quality, safe on RAM
BATCH_SIZE     = 256     # sentence-transformer batch size
N_CLUSTERS     = 2       # produces _baseline.pt and _baseline1.pt per language
RANDOM_SEED    = 42
MIN_LEN        = 15      # discard lines shorter than this
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def load_indic_data(json_path):
    """Read hindi/telugu lines from Data.json. Returns two lists."""
    print(f"\nLoading {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hindi_lines, telugu_lines = [], []
    for entry in data:
        h = entry.get("hindi", "").strip()
        t = entry.get("telugu", "").strip()
        if len(h) >= MIN_LEN:
            hindi_lines.append(h)
        if len(t) >= MIN_LEN:
            telugu_lines.append(t)

    print(f"  Hindi  lines available : {len(hindi_lines):,}")
    print(f"  Telugu lines available : {len(telugu_lines):,}")
    return hindi_lines, telugu_lines


def get_english_fallback(n=SAMPLE_SIZE):
    """
    Tries to stream Wikipedia English. Falls back to a curated sentence list
    so the script never fails even without internet / HuggingFace access.
    """
    print(f"\n--- Fetching English anchor data ---")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1",
                               split="train", streaming=True)
        samples = []
        for entry in tqdm(dataset, desc="Streaming Wikipedia", total=n):
            text = entry["text"].strip()
            if 60 < len(text) < 400 and not text.startswith(" ="):
                samples.append(text)
            if len(samples) >= n:
                break
        if samples:
            print(f"  Fetched {len(samples):,} English sentences from Wikipedia.")
            return samples
    except Exception as e:
        print(f"  Wikipedia stream failed: {e}")

    # ── Hard-coded diverse fallback (good enough for anchor quality) ──────────
    print("  Using built-in English fallback sentences.")
    base = [
        "The committee approved the annual budget after a lengthy discussion.",
        "Scientists discovered a new exoplanet in the habitable zone last month.",
        "She quickly finished her assignment and submitted it before the deadline.",
        "The stock market experienced significant volatility during the trading session.",
        "Local authorities are investigating the cause of the forest fire.",
        "The restaurant on the corner serves excellent pasta and wood-fired pizza.",
        "Children should have access to quality education regardless of their background.",
        "He apologized sincerely and promised it would not happen again.",
        "The new software update improves battery life by approximately fifteen percent.",
        "Researchers published their findings in a peer-reviewed journal last week.",
        "The highway was closed for three hours due to a multi-vehicle accident.",
        "She trained for months before competing in the national championship.",
        "Renewable energy sources are becoming increasingly cost-competitive worldwide.",
        "The documentary explores the impact of climate change on coastal communities.",
        "Engineers are working on a bridge that will connect the two islands.",
    ]
    # Replicate to reach SAMPLE_SIZE
    multiplied = (base * ((n // len(base)) + 1))[:n]
    random.shuffle(multiplied)
    return multiplied


def compute_clustered_anchors(model, sentences, lang_name, n_clusters=N_CLUSTERS):
    """
    Encodes sentences, runs KMeans, returns the L2-normalized cluster centroids.
    This is the KEY function — centroids are genuinely different vectors.
    """
    print(f"\n  Encoding {len(sentences):,} {lang_name} sentences ...")
    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=False,   # numpy array for KMeans
        normalize_embeddings=False,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"  Running KMeans (k={n_clusters}) on {lang_name} embeddings ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    kmeans.fit(embeddings)

    cluster_sizes = np.bincount(kmeans.labels_)
    print(f"  Cluster sizes: {cluster_sizes.tolist()}")

    centroids = []
    for i, center in enumerate(kmeans.cluster_centers_):
        vec = torch.tensor(center, dtype=torch.float32)
        vec = vec / torch.norm(vec)           # L2 normalize
        centroids.append(vec)
        print(f"  Cluster {i}: norm={torch.norm(vec).item():.4f}, "
              f"mean={vec.mean().item():.5f}, size={cluster_sizes[i]:,} sentences")

    return centroids


def save_anchors(centroids, lang_name, output_dir):
    """Saves centroids as {lang}_baseline.pt and {lang}_baseline1.pt etc."""
    suffixes = ["baseline", "baseline1", "baseline2", "baseline3"]
    saved = []
    for i, vec in enumerate(centroids):
        fname = f"{lang_name.lower()}_{suffixes[i]}.pt"
        fpath = os.path.join(output_dir, fname)
        torch.save(vec, fpath)
        saved.append(fpath)
        print(f"  Saved: {fpath}")
    return saved


def verify_anchors(saved_paths):
    """Quick sanity check: print similarity between the two anchors of each language."""
    print("\n=== ANCHOR VERIFICATION ===")
    # Group by language prefix
    from collections import defaultdict
    groups = defaultdict(list)
    for p in saved_paths:
        lang = os.path.basename(p).split("_")[0]
        groups[lang].append(p)

    for lang, paths in groups.items():
        vecs = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
        if len(vecs) >= 2:
            sim = torch.dot(vecs[0], vecs[1]).item()
            print(f"  {lang.capitalize()}: baseline0 vs baseline1 similarity = {sim:.6f}  "
                  f"({'⚠ TOO SIMILAR — clustering may have failed' if sim > 0.98 else '✓ Good separation'})")
        else:
            print(f"  {lang.capitalize()}: only 1 anchor (no comparison possible)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    print(f"Model : {MODEL_NAME}")
    print(f"Sample: {SAMPLE_SIZE:,} sentences per language")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")

    # 1. Load model
    print(f"\nLoading transformer model ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 2. Load source data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, JSON_FILE)
    hindi_pool, telugu_pool = load_indic_data(json_path)

    # 3. Sample (avoids RAM blowup and disk writes for sorted text)
    language_data = {
        "Hindi":   random.sample(hindi_pool,  min(len(hindi_pool),  SAMPLE_SIZE)),
        "Telugu":  random.sample(telugu_pool, min(len(telugu_pool), SAMPLE_SIZE)),
        "English": get_english_fallback(SAMPLE_SIZE),
    }

    print("\n=== SAMPLE SIZES ===")
    for lang, sents in language_data.items():
        print(f"  {lang}: {len(sents):,}")

    # 4. Generate anchors per language
    os.makedirs(os.path.join(script_dir, OUTPUT_DIR), exist_ok=True)
    output_dir_abs = os.path.join(script_dir, OUTPUT_DIR)

    all_saved = []
    for lang, sentences in language_data.items():
        print(f"\n{'='*50}")
        print(f"Processing: {lang}")
        print(f"{'='*50}")
        centroids = compute_clustered_anchors(model, sentences, lang, N_CLUSTERS)
        saved = save_anchors(centroids, lang, output_dir_abs)
        all_saved.extend(saved)

    # 5. Verify
    verify_anchors(all_saved)

    print("\n" + "="*50)
    print("ALL ANCHORS GENERATED SUCCESSFULLY.")
    print("Restart the Streamlit app to load the new anchors.")
    print("="*50)


if __name__ == "__main__":
    main()
