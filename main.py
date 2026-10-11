import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration shared by ALL four models
CFG = dict(
    vocab_size=4000,  # most frequent tokens kept, the rest become <unk>
    seq_len=20,  # context window (number of previous words)
    embed_dim=64,
    hidden_dim=128,
    dropout=0.3,
    epochs=10,
    batch_size=256,
    lr=2e-3,
    patience=3,  # early stopping on validation loss
    stride=1,  # sliding-window stride on the training split
    seed=42,
)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# ----------Dataset: The Adventures of Sherlock Holmes (Project Gutenberg #1661) ----------
DATA_URLS = [
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # fallback mirror path
]
DATA_PATH = "sherlock_holmes.txt"


def download(urls, path):
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
            print("Downloaded from", u)
            return
        except Exception as e:
            print("  failed:", u, e)
    raise SystemExit(
        "Could not download the dataset. Download it manually from "
        "https://www.gutenberg.org/ebooks/1661 (Plain Text UTF-8) and save it as "
        + path
    )


if not os.path.exists(DATA_PATH):
    download(DATA_URLS, DATA_PATH)
full = open(DATA_PATH, encoding="utf-8-sig").read()

# remove the Project Gutenberg licence header / footer, keep only the book itself
m_start = re.search(
    r"\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", full, re.S
)
m_end = re.search(r"\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG EBOOK", full)
raw = full[
    m_start.end() if m_start else 0 : m_end.start() if m_end else len(full)
].strip()
raw = raw.replace("\r\n", "\n")

print("\n--- Dataset characteristics ---")
print(f"Characters : {len(raw):,}")
print(f"Lines      : {raw.count(chr(10)) + 1:,}")
n_paragraphs = len(re.split(r"\n\s*\n", raw))
print(f"Paragraphs : {n_paragraphs:,}")
print(f"Words (whitespace split): {len(raw.split()):,}")
print("Sample:\n" + raw[1500:1900])


# - normalise curly quotes / dashes
# - lower-case
# - the book is hard-wrapped at ~70 chars, so single newlines are just spaces;
#   a blank line (paragraph break) becomes the special token <p>
# - split into words (apostrophes kept) and punctuation  . , ! ? ; : " - ( )
# - keep the top-N tokens, everything else -> <unk>
def tokenize(text):
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2014", " - ")
        .replace("--", " - ")
    )
    text = text.lower()
    text = re.sub(r"\n\s*\n", " <p> ", text)  # paragraph break
    text = text.replace("\n", " ")  # hard-wrapped line break -> space
    return re.findall(r"<p>|[a-z]+(?:'[a-z]+)*|[.,!?;:\"\-()]", text)


tokens = tokenize(raw)
counts = Counter(tokens)

itos = ["<unk>"] + [
    w for w, _ in counts.most_common(CFG["vocab_size"] - 1)
]  # index -> token
stoi = {w: i for i, w in enumerate(itos)}  # token -> index
V = len(itos)
ids = np.array([stoi.get(t, 0) for t in tokens], dtype=np.int64)

dataset_stats = dict(
    characters=len(raw),
    tokens=len(tokens),
    unique_tokens=len(counts),
    vocab_size=V,
    unk_percent=round(float(np.mean(ids == 0)) * 100, 2),
)
print("\n--- After preprocessing ---")
print(dataset_stats)
print("Top-10 tokens:", counts.most_common(10))
print("Tokens seen only once:", sum(1 for c in counts.values() if c == 1))
json.dump(dataset_stats, open(f"{OUT}/dataset_stats.json", "w"), indent=2)
json.dump(itos, open(f"{OUT}/vocab.json", "w"))

# token-frequency (Zipf) plot
freqs = sorted(counts.values(), reverse=True)
plt.figure(figsize=(5, 3.5))
plt.loglog(range(1, len(freqs) + 1), freqs)
plt.axvline(CFG["vocab_size"], color="r", ls="--", label="vocab cut-off")
plt.xlabel("rank")
plt.ylabel("frequency")
plt.title("Token frequency (Zipf)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/zipf.png", dpi=130)
plt.close()


# chronological 90 / 5 / 5 split (no leakage between splits)
# autoregressive pairs:  X[i] = w[i : i+L]   ->   y[i] = w[i+L]   (the next word)
n = len(ids)
a, b = int(0.90 * n), int(0.95 * n)
parts = {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def make_xy(arr, L, stride=1):
    starts = np.arange(0, len(arr) - L, stride)
    X = np.stack([arr[s : s + L] for s in starts])
    y = arr[starts + L]
    return torch.from_numpy(X), torch.from_numpy(y)


loaders = {}
print("\n--- Input-target sequences ---")
for name, arr in parts.items():
    X, y = make_xy(arr, CFG["seq_len"], CFG["stride"] if name == "train" else 1)
    g = torch.Generator()
    g.manual_seed(CFG["seed"])
    loaders[name] = DataLoader(
        TensorDataset(X, y),
        batch_size=CFG["batch_size"],
        shuffle=(name == "train"),
        generator=g,
    )
    print(f"{name:5s} tokens={len(arr):>8,}  X{tuple(X.shape)}  y{tuple(y.shape)}")

print("\nExample input  X:", " ".join(itos[i] for i in X[0].tolist()))
print("Example target y:", itos[y[0].item()])
