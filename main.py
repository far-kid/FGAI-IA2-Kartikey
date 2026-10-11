import json
import math
import os
import random
import re
import time
import urllib.request
from collections import Counter
from itertools import pairwise
from typing import TypedDict
from urllib.error import URLError

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TrainingConfig(TypedDict):
    vocab_size: int
    seq_len: int
    embed_dim: int
    hidden_dim: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    patience: int
    stride: int
    seed: int


# Configuration shared by ALL four models
CFG: TrainingConfig = {
    "vocab_size": 4000,  # most frequent tokens kept, the rest become <unk>
    "seq_len": 20,  # context window (number of previous words)
    "embed_dim": 64,
    "hidden_dim": 128,
    "dropout": 0.3,
    "epochs": 10,
    "batch_size": 256,
    "lr": 2e-3,
    "patience": 3,  # early stopping on validation loss
    "stride": 1,  # sliding-window stride on the training split
    "seed": 42,
}
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
        except (URLError, OSError) as e:
            print("  failed:", u, e)
    raise SystemExit(
        "Could not download the dataset. Download it manually from "
        "https://www.gutenberg.org/ebooks/1661 (Plain Text UTF-8) and save it as "
        + path
    )


if not os.path.exists(DATA_PATH):
    download(DATA_URLS, DATA_PATH)
with open(DATA_PATH, encoding="utf-8-sig") as f:
    full = f.read()

# remove the Project Gutenberg licence header / footer, keep only the book itself
m_start = re.search(
    r"\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", full, re.DOTALL
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

dataset_stats = {
    "characters": len(raw),
    "tokens": len(tokens),
    "unique_tokens": len(counts),
    "vocab_size": V,
    "unk_percent": round(float(np.mean(ids == 0)) * 100, 2),
}
print("\n--- After preprocessing ---")
print(dataset_stats)
print("Top-10 tokens:", counts.most_common(10))
print("Tokens seen only once:", sum(1 for c in counts.values() if c == 1))
with open(f"{OUT}/dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(dataset_stats, f, indent=2)
with open(f"{OUT}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(itos, f)

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
    if L <= 0 or stride <= 0:
        raise ValueError("Sequence length and stride must be positive.")
    if len(arr) <= L:
        raise ValueError(f"Split needs at least {L + 1} tokens; got {len(arr)}.")
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

example_x, example_y = loaders["train"].dataset[0]
print("\nExample input  X:", " ".join(itos[i] for i in example_x.tolist()))
print("Example target y:", itos[int(example_y.item())])

# =====================================================================================
# All four models share the SAME architecture skeleton:
#     Embedding(V, E) -> recurrent layer(H) -> Dropout -> Linear(-> V logits)
# Only the recurrent cell differs, so any performance difference comes from the cell type.
#
#   SimpleRNN : h_t = tanh(W_x x_t + W_h h_{t-1} + b)
#   BiRNN     : two SimpleRNNs, one reads the window left->right, one right->left;
#               their final hidden states are concatenated (output layer is 2H wide)
#   LSTM      : input / forget / output gates + cell state
#   GRU       : update / reset gates (fewer parameters than LSTM)
#
# The final hidden state(s) after reading the 20-word window summarise the context and are
# used to predict the next word.  (BiRNN is valid here because the window contains only
# words that are already known/generated; nothing from the future is used.)
KINDS = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]


class RecurrentLM(nn.Module):
    def __init__(self, kind, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.kind = kind
        self.bidirectional = kind == "BiRNN"
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if kind in ("SimpleRNN", "BiRNN"):
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                batch_first=True,
                nonlinearity="tanh",
                bidirectional=self.bidirectional,
            )
        elif kind == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif kind == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"unknown model kind: {kind}")
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, vocab_size)

    def forward(self, x):  # x: (batch, seq_len) of word indices
        _, h = self.rnn(self.embedding(x))  # h: final hidden state(s)
        if self.kind == "LSTM":
            h = h[0]  # LSTM returns (h_n, c_n); keep h_n
        if self.bidirectional:  # h: (2, batch, H) -> (batch, 2H)
            feat = torch.cat([h[-2], h[-1]], dim=1)
        else:  # h: (1, batch, H) -> (batch, H)
            feat = h[-1]
        return self.fc(self.dropout(feat))  # logits over the vocabulary


def build_model(kind):
    return RecurrentLM(kind, V, CFG["embed_dim"], CFG["hidden_dim"], CFG["dropout"]).to(
        DEVICE
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader):
    """Returns cross-entropy loss, perplexity (= exp(loss)), top-1 and top-5 next-word accuracy."""
    model.eval()
    total_loss, top1, top5, count = 0.0, 0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        top1 += (logits.argmax(dim=1) == yb).sum().item()
        top5 += (
            (logits.topk(min(5, logits.shape[1]), dim=1).indices == yb.unsqueeze(1))
            .any(dim=1)
            .sum()
            .item()
        )
        count += len(yb)
    if count == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    loss = total_loss / count
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "top1": top1 / count,
        "top5": top5 / count,
    }


def train_model(kind):
    """Train one model with the shared configuration; returns (model, history, test_metrics, info)."""
    set_seed(CFG["seed"])  # same initialisation seed for every model
    if CFG["epochs"] <= 0:
        raise ValueError("Training requires at least one epoch.")
    loaders["train"].generator.manual_seed(CFG["seed"])
    model = build_model(kind)
    n_params = count_params(model)
    print(f"\n=== {kind} | parameters: {n_params:,} | device: {DEVICE} ===")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=1
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "val_acc": [],
        "epoch_time": [],
    }
    best_val, best_state, bad_epochs = float("inf"), None, 0

    t_start = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        t_epoch = time.time()
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # gradient clipping for all models
            optimizer.step()
            running += loss.item() * len(yb)
            seen += len(yb)

        val = evaluate(model, loaders["val"])
        scheduler.step(val["loss"])
        history["train_loss"].append(running / seen)
        history["val_loss"].append(val["loss"])
        history["val_ppl"].append(val["perplexity"])
        history["val_acc"].append(val["top1"])
        history["epoch_time"].append(time.time() - t_epoch)
        print(
            f"[{kind}] epoch {epoch:02d} | train loss {running / seen:.3f} | val loss {val['loss']:.3f} "
            f"| val ppl {val['perplexity']:.1f} | val acc {val['top1']:.3f} | {history['epoch_time'][-1]:.1f}s"
        )

        if val["loss"] < best_val - 1e-4:  # keep the best-validation weights
            best_val, bad_epochs = val["loss"], 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= CFG["patience"]:
                print(f"[{kind}] early stopping at epoch {epoch}")
                break
    train_time = time.time() - t_start

    if best_state is None:
        raise RuntimeError(
            "Training produced no finite validation loss; no checkpoint was saved."
        )
    model.load_state_dict(best_state)  # restore best weights before testing
    test = evaluate(model, loaders["test"])
    print(
        f"[{kind}] TEST | loss {test['loss']:.3f} | perplexity {test['perplexity']:.1f} "
        f"| top-1 {test['top1']:.3f} | top-5 {test['top5']:.3f} | train time {train_time:.1f}s"
    )

    info = {
        "params": n_params,
        "epochs_run": len(history["val_loss"]),
        "train_time_s": train_time,
        "best_val_loss": best_val,
    }
    return model, history, test, info


# =====================================================================================
# Training
# Same data, split, shuffling, seed, optimiser, learning-rate schedule, gradient clipping,
# dropout, early-stopping rule and number of epochs for every model.
# NOTE: reported training time = wall-clock time of the whole training loop, including the
#       per-epoch validation pass (same for all models, so the comparison stays fair).
models, histories, test_results, run_info = {}, {}, {}, {}

if __name__ == "__main__":
    os.makedirs(f"{OUT}/models", exist_ok=True)
    for kind in KINDS:
        model, history, test, info = train_model(kind)
        models[kind], histories[kind], test_results[kind], run_info[kind] = (
            model,
            history,
            test,
            info,
        )
        torch.save(
            model.state_dict(), f"{OUT}/models/{kind}.pt"
        )  # best-validation weights

    # save everything so the analysis chunks (and the README) can use it
    with open(f"{OUT}/training_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": CFG,
                "histories": histories,
                "test": test_results,
                "info": run_info,
            },
            f,
            indent=2,
        )

    print("\n === Summary of training ===")
    for k in KINDS:
        t, i = test_results[k], run_info[k]
        print(
            f"{k:10s} params={i['params']:>9,}  epochs={i['epochs_run']:>2}  time={i['train_time_s']:7.1f}s  "
            f"test_loss={t['loss']:.3f}  ppl={t['perplexity']:7.1f}  top1={t['top1']:.3f}  top5={t['top5']:.3f}"
        )

# =====================================================================================
# Metrics: test loss, perplexity, top-1 / top-5 accuracy, training time, parameters,
#          convergence (epochs, curves) and over-fitting gap (validation - training loss).


def load_trained():
    """Reload weights and results saved by an earlier training run when needed."""
    global histories, test_results, run_info
    if models and test_results:
        return
    path = f"{OUT}/training_results.json"
    if not os.path.exists(path):
        raise SystemExit(
            "Run the training first: outputs/training_results.json not found."
        )
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    histories, test_results, run_info = saved["histories"], saved["test"], saved["info"]
    for kind in KINDS:
        m = build_model(kind)
        m.load_state_dict(torch.load(f"{OUT}/models/{kind}.pt", map_location=DEVICE))
        m.eval()
        models[kind] = m


load_trained()


rows = []
for kind in KINDS:
    t, i, h = test_results[kind], run_info[kind], histories[kind]
    rows.append(
        {
            "Model": kind,
            "Parameters": i["params"],
            "Epochs run": i["epochs_run"],
            "Train time (s)": round(i["train_time_s"], 1),
            "Sec/epoch": round(i["train_time_s"] / i["epochs_run"], 1),
            "Final train loss": round(h["train_loss"][-1], 3),
            "Best val loss": round(i["best_val_loss"], 3),
            "Test loss": round(t["loss"], 3),
            "Test perplexity": round(t["perplexity"], 1),
            "Top-1 acc": round(t["top1"], 4),
            "Top-5 acc": round(t["top5"], 4),
            # over-fitting gap at the best epoch: validation loss minus training loss of that epoch
            "Val-train gap": round(
                i["best_val_loss"] - h["train_loss"][int(np.argmin(h["val_loss"]))], 3
            ),
        }
    )
results_df = pd.DataFrame(rows)
results_df.to_csv(f"{OUT}/results.csv", index=False)


def df_to_markdown(df):
    head = (
        "| "
        + " | ".join(str(column) for column in df.columns)
        + " |\n|"
        + "---|" * len(df.columns)
        + "\n"
    )
    body = "\n".join(
        "| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)
    )
    return head + body


with open(f"{OUT}/results.md", "w", encoding="utf-8") as f:
    f.write(df_to_markdown(results_df))
print("\n=== Quantitative results (test split) ===")
print(results_df.to_string(index=False))

# ---------- Training curves ----------
COLORS = {
    "SimpleRNN": "#7f7f7f",
    "BiRNN": "#4c78a8",
    "LSTM": "#e45756",
    "GRU": "#54a24b",
}
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for kind in KINDS:
    h = histories[kind]
    ep = range(1, len(h["train_loss"]) + 1)
    ax[0].plot(ep, h["train_loss"], label=kind, color=COLORS[kind])
    ax[1].plot(ep, h["val_loss"], label=kind, color=COLORS[kind])
    ax[2].plot(ep, h["val_ppl"], label=kind, color=COLORS[kind])
    ax[3].plot(ep, h["val_acc"], label=kind, color=COLORS[kind])
for a_, title in zip(
    ax,
    [
        "Training loss",
        "Validation loss",
        "Validation perplexity",
        "Validation top-1 accuracy",
    ],
):
    a_.set_title(title)
    a_.set_xlabel("epoch")
    a_.grid(alpha=0.3)
    a_.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/curves.png", dpi=130)
plt.close()

# ---------- Bar charts: perplexity, accuracy, time, size ----------
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
bar_cols = [COLORS[k] for k in KINDS]
panels = [
    ("Test perplexity (lower is better)", "Test perplexity"),
    ("Training time (s)", "Train time (s)"),
    ("Parameters", "Parameters"),
]
for a_, (title, col) in zip(ax[:3], panels):
    vals = results_df[col].values
    a_.bar(KINDS, vals, color=bar_cols)
    a_.set_title(title)
    for i_, v in enumerate(vals):
        a_.text(
            i_,
            v,
            f"{v:,.0f}" if v > 1000 else f"{v:g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
x_pos = np.arange(len(KINDS))
w = 0.38
ax[3].bar(x_pos - w / 2, results_df["Top-1 acc"], w, label="top-1", color="#4c78a8")
ax[3].bar(x_pos + w / 2, results_df["Top-5 acc"], w, label="top-5", color="#f58518")
ax[3].set_xticks(x_pos)
ax[3].set_xticklabels(KINDS)
ax[3].set_title("Next-word accuracy")
ax[3].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/comparison_bars.png", dpi=130)
plt.close()

# ---------- Ranking printed from the measured numbers ----------
print("\n=== Ranking (from measured results) ===")
print(
    "Lowest test perplexity :",
    " < ".join(results_df.sort_values("Test perplexity")["Model"]),
)
print(
    "Highest top-1 accuracy :",
    " > ".join(results_df.sort_values("Top-1 acc", ascending=False)["Model"]),
)
print(
    "Fastest total training :",
    " < ".join(results_df.sort_values("Train time (s)")["Model"]),
)
print(
    "Smallest val-train gap :",
    " < ".join(results_df.sort_values("Val-train gap")["Model"]),
)
best_ppl = results_df.sort_values("Test perplexity").iloc[0]
print(
    f"\nBest perplexity: {best_ppl['Model']} ({best_ppl['Test perplexity']}); "
    f"random-guess perplexity for this vocabulary would be {V}."
)


# =====================================================================================
# Text generation, temperature analysis and conclusion
# =====================================================================================
# Autoregressive generation: feed the last `seq_len` words, turn the logits into a probability
# distribution with softmax(logits / T), SAMPLE the next word, append it, repeat.
#   T < 1  -> sharper distribution (safer, more repetitive)
#   T = 1  -> the model's own distribution
#   T > 1  -> flatter distribution (more diverse, more errors)
# The SAME seeds, the SAME random streams and the SAME settings are used for all four models.

TEMPS = [0.5, 1.0, 1.5]
GEN_LEN = 40  # words generated per sample
N_SAMPLES = 20  # samples per (model, seed, temperature) used for the statistics
SHOWN_SAMPLES = 2  # samples per combination written to generations.md
SEED_TEXTS = [
    "I had seen little of Holmes lately.",
    "\u201cMy dear Holmes,\u201d said I, \u201cthis is",
    "Holmes sat in his armchair and looked at",
]
SEEDS = [tokenize(t) for t in SEED_TEXTS]
for st, toks in zip(SEED_TEXTS, SEEDS):
    unk = [w for w in toks if w not in stoi]
    if unk:
        print(
            f"WARNING: seed '{st}' contains words outside the vocabulary: {unk} (they map to <unk>)"
        )

P_ID = stoi.get("<p>", 0)  # short seeds are left-padded with the paragraph-start token


@torch.no_grad()
def generate_batch(model, seed_tokens, n_words, temperature, n_samples, rng_seed):
    """Generate `n_samples` independent continuations of the seed in one batch. Returns list of id lists."""
    model.eval()
    L = CFG["seq_len"]
    ctx = [stoi.get(w, 0) for w in seed_tokens]
    if len(ctx) < L:
        ctx = [P_ID] * (L - len(ctx)) + ctx
    x = torch.tensor([ctx[-L:]] * n_samples, device=DEVICE)
    g = torch.Generator().manual_seed(
        rng_seed
    )  # identical random stream for every model
    out = []
    for _ in range(n_words):
        logits = model(x)
        logits[:, 0] = -1e9  # never generate <unk>
        probs = torch.softmax(logits / temperature, dim=1).cpu()
        nxt = torch.multinomial(probs, 1, generator=g)  # (n_samples, 1)
        out.append(nxt)
        x = torch.cat([x[:, 1:], nxt.to(DEVICE)], dim=1)
    return torch.cat(out, dim=1).tolist()


def detokenize(ids_list):
    text = " ".join(itos[i] for i in ids_list)
    text = text.replace(" <p> ", "\n\n").replace("<p> ", "\n\n").replace(" <p>", "\n\n")
    text = re.sub(r"[ \t]+([.,!?;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return text.strip()


train_ids = parts["train"].tolist()
train_bigrams = set(pairwise(train_ids))
train_trigrams = set(zip(train_ids, train_ids[1:], train_ids[2:]))


def gen_stats(gen_ids, last_seed_id):
    """Per-sample statistics of generated text.
    Diversity  : distinct1 / distinct2 = share of unique unigrams / bigrams.
    Repetition : rep_trigram = share of repeated trigrams; imm_repeat = share of w_t == w_(t+1).
    Coherence proxy : bigram_valid = share of adjacent word pairs that occur in the training text
                      (local grammatical plausibility; low values => word salad).
    Copying    : trigram_copy = share of generated trigrams that appear verbatim in the training text
                 (high at low temperature => memorisation / safe phrases)."""
    w = [i for i in gen_ids if i != P_ID]
    bi = list(pairwise(w))
    tri = list(zip(w, w[1:], w[2:]))
    seq = [last_seed_id] + gen_ids
    pairs = list(pairwise(seq))
    tri_all = list(zip(gen_ids, gen_ids[1:], gen_ids[2:]))
    return {
        "distinct1": len(set(w)) / max(len(w), 1),
        "distinct2": len(set(bi)) / max(len(bi), 1),
        "rep_trigram": 1 - len(set(tri)) / len(tri) if tri else 0.0,
        "imm_repeat": sum(a == b for a, b in bi) / max(len(bi), 1),
        "bigram_valid": sum(p in train_bigrams for p in pairs) / max(len(pairs), 1),
        "trigram_copy": sum(t in train_trigrams for t in tri_all)
        / max(len(tri_all), 1),
    }


md = [
    "# Generated text\n",
    (
        f"Same seeds, same random streams for all models. Each sample continues the seed by {GEN_LEN} words. "
        "Seed text is shown first; the continuation is in **bold**. "
        f"Statistics in `generation_summary.csv` are averaged over {N_SAMPLES} samples per combination.\n"
    ),
]
records = []
for si, (seed_text, seed_tokens) in enumerate(zip(SEED_TEXTS, SEEDS), start=1):
    md.append(f"\n## Seed {si}: {seed_text}\n")
    for T in TEMPS:
        md.append(f"\n### Temperature {T}\n")
        for kind in KINDS:
            samples = generate_batch(
                models[kind], seed_tokens, GEN_LEN, T, N_SAMPLES, rng_seed=1000 + si
            )
            last_id = stoi.get(seed_tokens[-1], 0)
            for s_ in samples:
                records.append(
                    {
                        "seed": si,
                        "temperature": T,
                        "model": kind,
                        **gen_stats(s_, last_id),
                    }
                )
            for s_ in samples[:SHOWN_SAMPLES]:
                md.append(f"- **{kind}**: {seed_text} **{detokenize(s_)}**\n")
with open(f"{OUT}/generations.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\nSaved generated samples to {OUT}/generations.md")


gen_df = pd.DataFrame(records)
gen_df.to_csv(f"{OUT}/generation_metrics_all_samples.csv", index=False)
METRICS = [
    "distinct1",
    "distinct2",
    "rep_trigram",
    "imm_repeat",
    "bigram_valid",
    "trigram_copy",
]
summary = (
    gen_df.groupby(["temperature", "model"])
    .agg({metric: "mean" for metric in METRICS})
    .round(3)
    .reset_index()
)
summary["model"] = pd.Categorical(summary["model"], KINDS, ordered=True)
summary = summary.sort_values(["temperature", "model"]).reset_index(drop=True)
summary.to_csv(f"{OUT}/generation_summary.csv", index=False)
print("\n=== Generation statistics (mean over seeds and samples) ===")
print(summary.to_string(index=False))


titles = {
    "distinct1": "Distinct-1 (diversity)",
    "distinct2": "Distinct-2 (diversity)",
    "rep_trigram": "Repeated-trigram rate (repetition, lower is better)",
    "imm_repeat": "Immediate word repeats (repetition)",
    "bigram_valid": "Bigram validity (coherence proxy, higher is better)",
    "trigram_copy": "Trigrams copied from training text",
}
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for a_, met in zip(axes.ravel(), METRICS):
    for kind in KINDS:
        s_ = summary[summary["model"] == kind]
        a_.plot(s_["temperature"], s_[met], "o-", label=kind, color=COLORS[kind])
    a_.set_title(titles[met], fontsize=10)
    a_.set_xlabel("temperature")
    a_.set_xticks(TEMPS)
    a_.grid(alpha=0.3)
    a_.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/temperature_effect.png", dpi=130)
plt.close()

# average over models: overall effect of temperature
overall = (
    summary.groupby("temperature").agg({metric: "mean" for metric in METRICS}).round(3)
)
print("\n=== Overall effect of temperature (mean over the four models) ===")
print(overall.to_string())


# generation_score = harmonic mean(coherence proxy, diversity) x (1 - repetition)
#   coherence proxy = bigram_valid, diversity = distinct2, repetition = rep_trigram
# It is only a helper: the final judgement must also use the generated text itself (generations.md).
def hmean(a, b):
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


summary["generation_score"] = [
    hmean(bigram_valid, distinct2) * (1 - rep_trigram)
    for bigram_valid, distinct2, rep_trigram in summary[
        ["bigram_valid", "distinct2", "rep_trigram"]
    ].itertuples(index=False, name=None)
]
score_tbl = summary.pivot(
    index="model", columns="temperature", values="generation_score"
).round(3)
score_tbl["best_T"] = score_tbl.idxmax(axis=1)
score_tbl["best_score"] = score_tbl[TEMPS].max(axis=1).round(3)
print("\n=== Generation score (higher is better) ===")
print(score_tbl.to_string())

rank_tbl = pd.DataFrame(
    {
        "perplexity_rank": results_df.set_index("Model")["Test perplexity"].rank(),
        "gen_score_rank": score_tbl["best_score"].rank(ascending=False),
        "time_rank": results_df.set_index("Model")["Train time (s)"].rank(),
    }
).loc[KINDS]
rank_tbl["combined(ppl+gen)"] = rank_tbl["perplexity_rank"] + rank_tbl["gen_score_rank"]
print("\n=== Ranks (1 = best) ===")
print(rank_tbl.to_string())
winner = rank_tbl["combined(ppl+gen)"].idxmin()
print(f"\nBest overall model by perplexity + generation score: {winner}")


readme_md = [
    "## Results (test split)\n",
    df_to_markdown(results_df),
    "\n\n## Generation statistics\n",
    df_to_markdown(summary.drop(columns=["generation_score"])),
    "\n\n## Generation score and best temperature per model\n",
    df_to_markdown(score_tbl.reset_index()),
    "\n\n## Ranks\n",
    df_to_markdown(rank_tbl.reset_index().rename(columns={"index": "model"})),
    f"\n\nBest overall model (perplexity + generation score): **{winner}**\n",
]
with open(f"{OUT}/summary_for_readme.md", "w", encoding="utf-8") as f:
    f.write("\n".join(readme_md))
print(f"Saved {OUT}/summary_for_readme.md  (tables to paste into README.md)")
