"""Sequential detector: a per-user LSTM over ordered transactions (Phase 3).

Hard fraud carries signal in the *sequence* of a user's transactions (Phase 1.1). This
module trains a many-to-many LSTM that labels each transaction as fraud/not, using the
same leakage-safe features and the same user-grouped split as the row-level models, then
reports per-tier recall (the comparison of interest is ``recall_hard`` vs the best
row-level model).

Robustness: the whole routine is wrapped so that if torch is missing or anything goes
wrong (degenerate folds, numerical issues), it returns ``{"skipped": True, "reason": ...}``
instead of breaking the benchmark. Transformers are intentionally not used.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from fraudlab.models.metrics import compute_metrics
from fraudlab.models.preprocessing import Design

MAX_LEN = 64
HIDDEN = 64
EPOCHS = 12
BATCH_SIZE = 256
LR = 1e-3


def _build_sequences(
    design: Design,
    X_dense: np.ndarray,
    row_indices: np.ndarray,
    max_len: int,
) -> list[dict[str, np.ndarray]]:
    """Group the given rows into per-user, time-ordered sequences (chunked to ``max_len``)."""
    groups = design.groups
    ts = design.timestamps
    y = design.y

    by_user: dict[str, list[int]] = {}
    for pos in row_indices:
        by_user.setdefault(groups[pos], []).append(int(pos))

    sequences: list[dict[str, np.ndarray]] = []
    for _user, positions in by_user.items():
        positions.sort(key=lambda p: ts[p])
        for start in range(0, len(positions), max_len):
            chunk = positions[start : start + max_len]
            arr = np.asarray(chunk, dtype=np.int64)
            sequences.append(
                {
                    "feat": X_dense[arr],
                    "label": y[arr].astype(np.float32),
                    "pos": arr,
                }
            )
    return sequences


def _pad_batch(seqs: list[dict[str, np.ndarray]], n_features: int):
    """Pad a list of sequences into (feat, label, mask, pos) padded arrays."""
    lengths = [len(s["label"]) for s in seqs]
    max_len = max(lengths)
    b = len(seqs)
    feat = np.zeros((b, max_len, n_features), dtype=np.float32)
    label = np.zeros((b, max_len), dtype=np.float32)
    mask = np.zeros((b, max_len), dtype=np.float32)
    pos = np.full((b, max_len), -1, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = lengths[i]
        feat[i, :L] = s["feat"]
        label[i, :L] = s["label"]
        mask[i, :L] = 1.0
        pos[i, :L] = s["pos"]
    return feat, label, mask, pos


def run_lstm(
    design: Design,
    X_dense: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    random_state: int = 42,
    max_len: int = MAX_LEN,
    epochs: int = EPOCHS,
) -> dict[str, Any]:
    """Train the LSTM and return its test metrics, or a skip record on any failure."""
    if importlib.util.find_spec("torch") is None:
        return {"skipped": True, "reason": "torch not installed"}

    try:
        import torch
        from torch import nn

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        torch.use_deterministic_algorithms(False)

        y = design.y
        if int(y[train_idx].sum()) == 0 or int(y[test_idx].sum()) == 0:
            return {"skipped": True, "reason": "a fold has no fraud rows"}

        n_features = X_dense.shape[1]
        train_seqs = _build_sequences(design, X_dense, train_idx, max_len)
        test_seqs = _build_sequences(design, X_dense, test_idx, max_len)
        if not train_seqs or not test_seqs:
            return {"skipped": True, "reason": "no sequences could be built"}

        pos = int(y[train_idx].sum())
        neg = int((y[train_idx] == 0).sum())
        pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], dtype=torch.float32)

        class LSTMTagger(nn.Module):
            def __init__(self, in_dim: int, hidden: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out).squeeze(-1)

        model = LSTMTagger(n_features, HIDDEN)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        rng = np.random.default_rng(random_state)
        model.train()
        for _epoch in range(epochs):
            order = rng.permutation(len(train_seqs))
            for start in range(0, len(order), BATCH_SIZE):
                batch = [train_seqs[i] for i in order[start : start + BATCH_SIZE]]
                feat, label, mask, _ = _pad_batch(batch, n_features)
                xb = torch.from_numpy(feat)
                yb = torch.from_numpy(label)
                mb = torch.from_numpy(mask)
                opt.zero_grad()
                logits = model(xb)
                loss = (loss_fn(logits, yb) * mb).sum() / mb.sum().clamp(min=1.0)
                loss.backward()
                opt.step()

        # Inference over test sequences -> per-row scores aligned to test_idx.
        model.eval()
        pos_to_score: dict[int, float] = {}
        with torch.no_grad():
            for start in range(0, len(test_seqs), BATCH_SIZE):
                batch = test_seqs[start : start + BATCH_SIZE]
                feat, _label, mask, posb = _pad_batch(batch, n_features)
                probs = torch.sigmoid(model(torch.from_numpy(feat))).numpy()
                for i in range(len(batch)):
                    valid = mask[i] > 0
                    for p, sc in zip(posb[i][valid], probs[i][valid]):
                        pos_to_score[int(p)] = float(sc)

        scores = np.array([pos_to_score.get(int(p), 0.0) for p in test_idx], dtype=float)
        labels = (scores >= 0.5).astype(int)
        metrics = compute_metrics(y[test_idx], scores, labels, design.tiers[test_idx])
        metrics["skipped"] = False
        metrics["kind"] = "sequential"
        return metrics
    except Exception as exc:  # pragma: no cover - defensive skip path
        return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
