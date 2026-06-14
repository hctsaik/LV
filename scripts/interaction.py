"""Pure interaction logic for the Streamlit GUI — no streamlit imports.

Ported/adapted from VIX's framework-agnostic ``core/`` pattern: every
function here is unit-testable without a browser or a Streamlit session.
"""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import hnswlib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

from signal_strength import SIGNAL_NONE, SIGNAL_OBVIOUS


def parse_folder_paths(text: str) -> list[Path]:
    """Parse a newline-separated folder list into Path objects.

    One path per line; blank/whitespace-only lines are ignored; surrounding
    whitespace (including trailing ``\\r`` from CRLF input) is stripped.
    No existence check is performed — the caller validates.
    """
    if not text:
        return []
    paths = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def build_nn_index(emb_matrix: np.ndarray) -> hnswlib.Index:
    """Build a cosine HNSW index (hnswlib) over ``emb_matrix`` (N, D).

    The interactive query layer for F3 image-query and F7 text-query.
    HNSW is approximate by design; with these parameters (M=16,
    ef_construction=200, query ef >= 4k) recall is effectively 100% at the
    few-thousand-image scale this tool targets. Batch statistics
    (outlier-ness, label disagreement, dup radius scan) stay on sklearn
    exact search.
    """
    emb = np.ascontiguousarray(np.asarray(emb_matrix), dtype=np.float32)
    n, dim = emb.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max(n, 1), ef_construction=200, M=16,
                     random_seed=42)
    if n:
        index.add_items(emb, np.arange(n))
    return index


def _knn_query(nn_index: hnswlib.Index, vec: np.ndarray, k: int):
    nn_index.set_ef(max(64, k * 4))
    labels, dists = nn_index.knn_query(
        np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1), k=k)
    # float32 rounding can give ~-1e-7 for identical vectors — clamp
    return labels[0], np.maximum(dists[0], 0.0)


def find_similar_indices(
    emb_matrix: np.ndarray,
    query_idx: int,
    k: int = 9,
    nn_index: hnswlib.Index | None = None,
) -> tuple[list[int], list[float]]:
    """Return (indices, cosine_distances) of the k nearest neighbours to
    ``emb_matrix[query_idx]``, EXCLUDING the query itself.

    Results are sorted by ascending distance. ``k`` is clamped to N-1.
    Raises IndexError for an out-of-range ``query_idx``.
    """
    emb_matrix = np.asarray(emb_matrix)
    n = len(emb_matrix)
    if not 0 <= query_idx < n:
        raise IndexError(f"query_idx {query_idx} out of range for {n} embeddings")
    k = max(0, min(k, n - 1))
    if k == 0:
        return [], []
    if nn_index is None:
        nn_index = build_nn_index(emb_matrix)
    idx, dist = _knn_query(nn_index, emb_matrix[query_idx], min(k + 1, n))
    out_idx, out_dist = [], []
    for i, d in zip(idx, dist):
        if int(i) == query_idx:
            continue
        out_idx.append(int(i))
        out_dist.append(float(d))
    # if the query wasn't among the k+1 (duplicate rows), trim to k
    return out_idx[:k], out_dist[:k]


def find_similar_to_vector(
    emb_matrix: np.ndarray,
    query_vec: np.ndarray,
    k: int = 9,
    nn_index: hnswlib.Index | None = None,
) -> tuple[list[int], list[float]]:
    """Return (indices, cosine_distances) of the k nearest rows to an
    EXTERNAL query vector — e.g. a Chinese-CLIP text embedding (F7).

    No self-exclusion (the query is not a library row). ``k`` is clamped
    to N. The query's dimensionality must match the matrix.
    """
    emb_matrix = np.asarray(emb_matrix)
    n = len(emb_matrix)
    if n == 0:
        return [], []
    query_vec = np.asarray(query_vec).reshape(-1)
    if query_vec.shape[0] != emb_matrix.shape[1]:
        raise ValueError(
            f"query dim {query_vec.shape[0]} != embedding dim {emb_matrix.shape[1]}")
    k = max(1, min(k, n))
    if nn_index is None:
        nn_index = build_nn_index(emb_matrix)
    idx, dist = _knn_query(nn_index, query_vec, k)
    return [int(i) for i in idx], [float(d) for d in dist]


def compute_outlier_scores(
    candidates: np.ndarray,
    reference: np.ndarray,
    k: int = 5,
    candidates_in_reference: bool = False,
) -> np.ndarray:
    """Outlier-ness = mean cosine distance from each candidate row to its
    k nearest neighbours in ``reference`` (label-free novelty).

    Higher = more unlike the reference set. NOT a probability or an error
    verdict — the UI must disclaim this. ``k`` is clamped to the reference
    size. When ``candidates_in_reference`` is True the closest neighbour of
    each candidate (its own row) is dropped before averaging.
    """
    candidates = np.asarray(candidates)
    reference = np.asarray(reference)
    if len(candidates) == 0:
        return np.zeros(0, dtype=float)
    extra = 1 if candidates_in_reference else 0
    k_eff = max(1, min(k + extra, len(reference)))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(reference)
    dist, _ = nn.kneighbors(candidates, n_neighbors=k_eff)
    if candidates_in_reference and dist.shape[1] > 1:
        dist = dist[:, 1:]
    return dist.mean(axis=1)


_CURATION_HEADER = ["ts", "reason", "n", "filenames", "sha256s"]


def curation_log_csv(entries: Sequence[dict]) -> str:
    """Export the curation log (selection + free-text reason) to CSV.

    One row per logged selection: timestamp, the curator's reason, how
    many images, and the filenames / sha256s (content-addressed) so the
    record is auditable and transferable to a teammate."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_CURATION_HEADER)
    for e in entries:
        items = e.get("items", [])
        w.writerow([
            e.get("ts", ""), e.get("reason", ""), e.get("n", len(items)),
            " ".join(it.get("filename", "") for it in items),
            " ".join(it.get("sha256", "") for it in items),
        ])
    return buf.getvalue()


def match_shas_to_indices(
    shas: Sequence[str], sha_to_index: dict[str, int],
) -> list[int]:
    """Map a logged selection's content hashes back to current record
    indices (re-select『回到上週的選取』). Hashes not present in the
    current run are silently skipped, so a log made on one dataset
    degrades gracefully on another. De-duplicated, order preserved."""
    seen: set[int] = set()
    out: list[int] = []
    for s in shas:
        i = sha_to_index.get(s)
        if i is not None and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def select_gray_zone(scores: np.ndarray, k: int) -> list[int]:
    """Gray-zone review queue (§3): the k most ambiguous items by score
    (e.g. kNN label-disagreement), highest first. ``k`` clamped to N."""
    s = np.asarray(scores)
    if len(s) == 0 or k <= 0:
        return []
    return [int(i) for i in np.argsort(s)[::-1][:min(k, len(s))]]


def gray_zone_summary(
    scores: np.ndarray, *, thr: float = 0.5, hi: float = 0.6, lo: float = 0.3,
) -> dict:
    """Backlog stats for the gray-zone queue from per-sample label-disagreement
    scores — so the UI can show "how much actually needs auditing" instead of a
    fixed top-N. ``thr`` is the gray-band cutoff; ``hi``/``lo`` split severity.

    Returns {n_total, n_gray (≥thr), pct_gray, high (≥hi), mid (lo–hi),
    low (>0–lo), clean (==0)}.
    """
    d = np.asarray(scores, dtype=float)
    n = int(d.size)
    n_gray = int((d >= thr).sum())
    return {
        "n_total": n,
        "n_gray": n_gray,
        "pct_gray": round(100.0 * n_gray / n, 1) if n else 0.0,
        "high": int((d >= hi).sum()),
        "mid": int(((d >= lo) & (d < hi)).sum()),
        "low": int(((d > 0) & (d < lo)).sum()),
        "clean": int((d == 0).sum()),
    }


def nearest_anchor(
    embeddings: np.ndarray, idx: int, anchor_indices: Sequence[int],
) -> tuple[int | None, float]:
    """Closest anchor to ``idx`` by cosine distance (which 明確是/明確否
    example this gray-zone item sits nearest to). Returns (anchor_idx, d);
    (None, inf) when there are no anchors."""
    anchors = [a for a in anchor_indices if a != idx]
    if not anchors:
        return None, float("inf")
    emb = np.asarray(embeddings, dtype=float)
    q = emb[idx] / (np.linalg.norm(emb[idx]) + 1e-12)
    best, best_d = None, float("inf")
    for a in anchors:
        v = emb[a] / (np.linalg.norm(emb[a]) + 1e-12)
        d = float(1.0 - q @ v)
        if d < best_d:
            best, best_d = a, d
    return best, best_d


_GRAY_DECISION_HEADER = [
    "path", "soft_label", "confidence", "anchor", "reason",
    "proposer", "approver", "status",
]


def gray_decision_csv(decisions: Sequence[dict]) -> str:
    """Serialize confirmed gray-zone decisions to CSV — the four required
    provenance fields (who confirmed, anchor compared, soft label, reason)
    plus status, so an approved decision is auditable downstream."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_GRAY_DECISION_HEADER)
    for d in decisions:
        w.writerow([d.get(k, "") for k in _GRAY_DECISION_HEADER])
    return buf.getvalue()


def farthest_point_sampling(
    embeddings: np.ndarray,
    n: int,
    seed_indices: Sequence[int] | None = None,
) -> list[int]:
    """k-center greedy / farthest-point sampling (F6 diversity selection).

    Iteratively pick the row whose cosine distance to the already-covered
    set is largest — the maximally-diverse subset to label next. With
    ``seed_indices`` (already-labeled rows) the picks COVER the gaps:
    points far from every seed are chosen first, which is the active-
    learning use (complements the heatmap's per-cell candidate mining).
    Without seeds it starts from the most peripheral point.

    Returns up to ``n`` indices in pick order (most diverse first), never
    including a seed.
    """
    emb = np.asarray(embeddings, dtype=float)
    N = len(emb)
    if N == 0 or n <= 0:
        return []
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    excluded: set[int] = set()
    selected: list[int] = []
    seeds = [i for i in (seed_indices or []) if 0 <= i < N]
    if seeds:
        min_d = (1.0 - norm @ norm[seeds].T).min(axis=1)
        excluded.update(seeds)
    else:
        centroid = norm.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        first = int(np.argmax(1.0 - norm @ centroid))
        selected.append(first)
        excluded.add(first)
        min_d = 1.0 - norm @ norm[first]
    n = min(n, N - len(excluded) + (0 if seeds else 1))
    while len(selected) < n:
        md = min_d.copy()
        if excluded:
            md[list(excluded)] = -1.0
        nxt = int(np.argmax(md))
        if md[nxt] < 0:
            break
        selected.append(nxt)
        excluded.add(nxt)
        min_d = np.minimum(min_d, 1.0 - norm @ norm[nxt])
    return selected


def load_scores_csv(csv_path: Path) -> dict[str, tuple[float, float | None]]:
    """Optional detection-score ingestion for the escape card (N4 gate).

    Reads a ``scores.csv`` with columns ``filename,score[,threshold]`` →
    ``{filename: (score, threshold|None)}``. This is how a detection model's
    output enters LV without LV running the detector. Missing file or
    unparseable rows yield an empty / partial map (the card degrades to
    embedding-only attribution).
    """
    csv_path = Path(csv_path)
    out: dict[str, tuple[float, float | None]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or row.get("file") or "").strip()
            if not name:
                continue
            try:
                score = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
            thr = row.get("threshold")
            try:
                thr_v = float(thr) if thr not in (None, "") else None
            except ValueError:
                thr_v = None
            out[name] = (score, thr_v)
    return out


def load_predictions_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Ingest a detection model's per-image predictions for the evaluation gate.

    Generalises ``load_scores_csv`` from one score/image to many boxes/image,
    so LV can do IoU evaluation against (consensus) ground truth WITHOUT
    running the detector itself — it only ingests the detector's output.

    Reads a ``predictions.csv`` whose header (case-insensitive, order-free)
    has ``filename, class, cx, cy, w, h[, score]``; the box is normalized YOLO
    format (cx,cy,w,h ∈ [0,1]) to match the GT label files. ``class`` may be a
    name or id and is kept as a string. Missing file / unparseable rows are
    skipped → partial/empty map.

    Returns ``{filename: [{cls, cx, cy, w, h, score}, …]}``.
    """
    csv_path = Path(csv_path)
    out: dict[str, list[dict]] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            r = {(k or "").strip().lower(): (v or "").strip()
                 for k, v in row.items()}
            name = r.get("filename") or r.get("file") or r.get("image")
            cls = r.get("class") or r.get("label") or r.get("class_id")
            if not name or not cls:
                continue
            try:
                cx, cy, w, h = (float(r[k]) for k in ("cx", "cy", "w", "h"))
            except (KeyError, ValueError):
                continue
            score_s = r.get("score") or r.get("confidence") or ""
            try:
                score = float(score_s) if score_s else 1.0
            except ValueError:
                score = 1.0
            out.setdefault(name, []).append(
                {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h, "score": score})
    return out


def neighbor_hit_density(
    emb_matrix: np.ndarray,
    query_idx: int,
    radius: float,
    nn_index=None,
) -> int:
    """N2 signal: how many OTHER rows fall within cosine ``radius`` of the
    query (excluding itself). Low density = the model was shown few
    similar examples → sample scarcity candidate."""
    emb = np.asarray(emb_matrix)
    n = len(emb)
    if not 0 <= query_idx < n or n < 2:
        return 0
    q = emb[query_idx]
    qn = q / (np.linalg.norm(q) + 1e-12)
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    dist = 1.0 - en @ qn
    return int(np.sum(dist <= radius)) - 1  # drop self (distance 0)


def neighbor_label_entropy(
    emb_matrix: np.ndarray,
    labels: Sequence[str],
    query_idx: int,
    k: int = 20,
) -> float:
    """N3 signal: normalized Shannon entropy ∈ [0,1] of the k nearest
    neighbours' labels (self excluded). High = neighbours disagree on the
    label → standard-drift / labeling-dispute candidate."""
    emb = np.asarray(emb_matrix)
    n = len(emb)
    if not 0 <= query_idx < n or n < 2:
        return 0.0
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(emb)
    _, idx = nn.kneighbors(emb[query_idx:query_idx + 1], n_neighbors=min(k_eff + 1, n))
    neigh = [int(j) for j in idx[0] if int(j) != query_idx][:k_eff]
    if not neigh:
        return 0.0
    arr = np.asarray(labels, dtype=object)[neigh]
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    ent = -float(np.sum(p * np.log(p)))
    max_ent = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_ent) if max_ent > 0 else 0.0


# escape attribution classes (defect-mechanisms decision tree A–E)
ESCAPE_A = "A 標準漂移"        # neighbours disagree on label (N3 high)
ESCAPE_B = "B 樣本稀缺"        # few similar training examples (N2 low)
ESCAPE_C = "C 邊界擦邊"        # model score sits next to the threshold (N4)
ESCAPE_D = "D 新型態"          # no neighbours + outlier (N2 zero, novelty high)
ESCAPE_REVIEW = "需人工覆核"    # signals insufficient to attribute


def attribute_escape(
    hit_density: int,
    label_entropy: float,
    outlier_pct: float,
    score: float | None = None,
    threshold: float | None = None,
    entropy_thr: float = 0.8,
    density_thr: int = 3,
) -> dict:
    """Preliminary escape attribution from embedding signals (+ optional
    model score). Honest by construction: N0/N1 need a human and N4 needs
    a detection score, so without a score this only separates A/B/D and
    otherwise defers to 需人工覆核.

    Returns {class, confidence (0-1), reasons[list]}.
    """
    reasons: list[str] = []
    if label_entropy >= entropy_thr:
        reasons.append(f"鄰居標籤分歧高（熵 {label_entropy:.2f} ≥ {entropy_thr}）")
        return {"class": ESCAPE_A, "confidence": round(min(label_entropy, 1.0), 2),
                "reasons": reasons}
    if hit_density == 0 and outlier_pct >= 0.9:
        reasons.append(f"訓練集無相似鄰居，且離群度居前 {(1 - outlier_pct) * 100:.0f}%")
        return {"class": ESCAPE_D, "confidence": round(outlier_pct, 2),
                "reasons": reasons}
    if score is not None and threshold is not None:
        margin = abs(score - threshold)
        rel = margin / (abs(threshold) + 1e-9)
        if rel <= 0.1:
            reasons.append(f"模型分數貼近閾值（|{score:.3f}−{threshold:.3f}| 相對 {rel*100:.0f}%）")
            return {"class": ESCAPE_C, "confidence": round(1 - rel, 2), "reasons": reasons}
    if hit_density <= density_thr:
        reasons.append(f"訓練集相似鄰居稀少（{hit_density} ≤ {density_thr}）")
        return {"class": ESCAPE_B, "confidence": round(1 - hit_density / (density_thr + 1), 2),
                "reasons": reasons}
    reasons.append("embedding 訊號不足以歸因（鄰居充足且標籤一致）；"
                   "需 N0 品質/N1 定義/N4 分數判定")
    return {"class": ESCAPE_REVIEW, "confidence": 0.3, "reasons": reasons}


# ── three-orthogonal-signal root-cause diagnosis (H1–H5) ────────────────
# (vision-judgment-boundary framework §3 / defect-mechanisms decision tree)
# H0 is the 桶① physical-detectability gate that sits in FRONT of H1–H5
# (signal_strength.py): the original three-signal framework had no axis for
# "is the defect's signal even in the pixels", so a never-captured defect
# would fall through to H1 and be told "補資料大概率有效" — the exact wrong
# advice. H0 short-circuits that case.
CAUSE_H0 = "H0 物理天花板"
CAUSE_H1 = "H1 覆蓋缺口"
CAUSE_H2 = "H2 定義歧義"
CAUSE_H3 = "H3 標籤雜訊"
CAUSE_H4 = "H4 容量/特徵限制"
CAUSE_H5 = "H5 分布外 OOD"

_CAUSE_ACTION = {
    CAUSE_H0: "訊號未進資料（桶①）：ROI 內沒有高於背景的可偵測證據——補資料／調閾值／"
              "換模型都無效，要動成像鏈（解析度／放大／打光／曝光）。這是粗判，"
              "務必用真實 escape 就地校準訊號門檻後再下定論。",
    CAUSE_H1: "補這一格的資料——『本該判對只是資料太少』，補了大概率有效。",
    CAUSE_H2: "定義問題，非資料問題：凍結為灰帶、送仲裁人、更新定義書（補資料不會收斂，標籤互相矛盾）。",
    CAUSE_H3: "稽核這一帶的訓練標籤、查混淆變數——模型可能其實是對的、GT 錯了。",
    CAUSE_H4: "換架構 / 加特徵 / 上專家模型——落在密集格仍學不會，是容量天花板，補資料幫助有限。",
    CAUSE_H5: "看似 in-distribution 實則新樣態：收新樣態資料，並檢查 embedding 分不分得開。",
}
_CAUSE_ADD_DATA = {
    CAUSE_H0: "無效（訊號不在資料裡，需改成像）",
    CAUSE_H1: "有效（補這格）",
    CAUSE_H2: "無效（是定義問題，補資料不收斂）",
    CAUSE_H3: "先別補（先稽核標籤，GT 可能才錯）",
    CAUSE_H4: "幫助有限（要換架構）",
    CAUSE_H5: "需收『新樣態』資料",
}


def diagnose_root_cause(
    s1_consistency: float,
    s2_density: float,
    s3_entropy: float,
    *,
    signal_level: str | None = None,
    consistency_thr: float = 0.75,
    density_thr: float = 3,
    entropy_thr: float = 0.5,
) -> dict:
    """Cross-locate why a misjudged sample failed, from three ORTHOGONAL
    signals (none alone is enough), gated by a 桶① physical-detectability
    screen:

      S0 ``signal_level`` — is the defect's signal even in the pixels?
        (signal_strength.classify_signal). SIGNAL_NONE ("確無") overrides
        everything → H0 物理天花板: no data can help. None = not measured.
      S1 ``s1_consistency`` — concept ambiguity (HUMAN consistency, e.g.
        the quiz / gauge-R&R score). Low = experts themselves disagree.
      S2 ``s2_density`` — local data coverage (embedding neighbour count
        within the N2 radius). Sparse = a coverage gap.
      S3 ``s3_entropy`` — model uncertainty (softmax entropy / 1−margin).
        High = the model itself hesitates; low = confidently wrong.

    Returns {cause, action, add_data, s1_low, s2_sparse, s3_high,
    signal_level, caveat}. The headline ``add_data`` answers the founding
    question — does adding data actually help — which only H1 (and H5, for
    new regimes) does. ``caveat`` is non-empty when an H1 "補資料有效"
    verdict rests on UNVERIFIED detectability (signal not confirmed
    present), so the advice is never trusted blindly without the 桶① check.
    """
    s1_low = s1_consistency < consistency_thr
    s2_sparse = s2_density <= density_thr
    s3_high = s3_entropy >= entropy_thr
    if signal_level == SIGNAL_NONE:  # 桶①: signal not in the pixels at all
        cause = CAUSE_H0
    elif s1_low:                     # humans disagree → it's a spec problem
        cause = CAUSE_H2
    elif s3_high:                    # model hesitates
        cause = CAUSE_H1 if s2_sparse else CAUSE_H4
    else:                            # confidently wrong
        cause = CAUSE_H5 if s2_sparse else CAUSE_H3

    caveat = ""
    if cause == CAUSE_H1 and signal_level != SIGNAL_OBVIOUS:
        caveat = (
            f"H1 假設『缺陷可偵測、只是資料太少』，但訊號強度為 "
            f"{signal_level or '未量測'}——尚未確認訊號真的在資料裡。"
            "若其實落在桶①（物理天花板），補再多資料也無效；建議先量訊號強度再投資補樣。"
        )
    return {
        "cause": cause,
        "action": _CAUSE_ACTION[cause],
        "add_data": _CAUSE_ADD_DATA[cause],
        "s1_low": s1_low, "s2_sparse": s2_sparse, "s3_high": s3_high,
        "signal_level": signal_level, "caveat": caveat,
    }


def compute_label_disagreement(
    embeddings: np.ndarray,
    labels: Sequence[str],
    k: int = 5,
) -> np.ndarray:
    """Label audit (F5): for each row, the fraction of its k nearest
    neighbours (self excluded) carrying a DIFFERENT label.

    0 = neighbourhood agrees, 1 = neighbourhood disagrees. This is a
    neighbourhood statistic, NOT a mislabel verdict — the UI must keep the
    honest framing. ``k`` is clamped to N-1; with fewer than 2 rows (or a
    single class) every score is 0.
    """
    embeddings = np.asarray(embeddings)
    labels = list(labels)
    n = len(labels)
    if n < 2:
        return np.zeros(n, dtype=float)
    k_eff = max(1, min(k, n - 1))
    nn = NearestNeighbors(metric="cosine")
    nn.fit(embeddings)
    # +1 so we can drop each row's own entry (byte-identical duplicates may
    # shuffle who comes first, so drop by index, not by position)
    _, idx = nn.kneighbors(embeddings, n_neighbors=min(k_eff + 1, n))
    arr = np.asarray(labels, dtype=object)
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        neigh = [j for j in idx[i] if j != i][:k_eff]
        if neigh:
            scores[i] = float(np.mean(arr[neigh] != arr[i]))
    return scores


def hamming_distance_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex hash strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _filter_pair(i: int, j: int, splits: Sequence[str] | None,
                 cross_split_only: bool) -> bool:
    if not cross_split_only:
        return True
    return splits is not None and splits[i] != splits[j]


def find_duplicate_pairs_phash(
    phashes: Sequence[str | None],
    max_hamming: int = 4,
    splits: Sequence[str] | None = None,
    cross_split_only: bool = False,
    max_pairs: int = 200,
) -> list[tuple[int, int, int]]:
    """Duplicate candidates (F4) by perceptual hash.

    Returns (i, j, hamming) with i < j, sorted by distance then indices,
    capped at ``max_pairs``. None hashes (unreadable images) are skipped.
    With cross_split_only=True only pairs spanning different splits are
    returned — i.e. train/val leakage candidates. O(N²) in vectorized
    chunks; fine for the few-thousand-image datasets this tool targets.
    """
    idx = [i for i, h in enumerate(phashes) if h]
    if len(idx) < 2:
        return []
    vals = np.array([np.uint64(int(phashes[i], 16)) for i in idx], dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for a in range(len(idx) - 1):
        xor = (vals[a] ^ vals[a + 1:]).astype(np.uint64)
        dists = np.unpackbits(xor.view(np.uint8)).reshape(len(xor), -1).sum(axis=1)
        for off in np.nonzero(dists <= max_hamming)[0]:
            i, j = idx[a], idx[a + 1 + off]
            if _filter_pair(i, j, splits, cross_split_only):
                pairs.append((i, j, int(dists[off])))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    return pairs[:max_pairs]


def find_duplicate_pairs_embedding(
    embeddings: np.ndarray,
    max_distance: float = 0.05,
    splits: Sequence[str] | None = None,
    cross_split_only: bool = False,
    max_pairs: int = 200,
) -> list[tuple[int, int, float]]:
    """Duplicate candidates (F4) by embedding cosine distance.

    Semantic near-duplicates that survive resizing/re-encoding, which
    phash misses. Same return contract as the phash variant.
    """
    embeddings = np.asarray(embeddings)
    if len(embeddings) < 2:
        return []
    nn = NearestNeighbors(metric="cosine", radius=max_distance)
    nn.fit(embeddings)
    dists, idxs = nn.radius_neighbors(embeddings)
    pairs: list[tuple[int, int, float]] = []
    for i, (ds, js) in enumerate(zip(dists, idxs)):
        for d, j in zip(ds, js):
            if j <= i:
                continue
            if _filter_pair(i, int(j), splits, cross_split_only):
                pairs.append((i, int(j), float(d)))
    pairs.sort(key=lambda p: (p[2], p[0], p[1]))
    return pairs[:max_pairs]


def selection_points_to_indices(points: list[dict]) -> list[int]:
    """Extract global record indices from a Streamlit plotly selection.

    ``points`` is ``event.selection.points``; each point carries our global
    index in ``customdata`` (either ``[i]`` or scalar ``i``). Missing/None
    customdata entries are skipped. Returns a de-duplicated list preserving
    first-seen order, values as plain python ints.
    """
    seen: set[int] = set()
    out: list[int] = []
    for pt in points or []:
        cd = pt.get("customdata")
        if cd is None:
            continue
        if isinstance(cd, (list, tuple)):
            if not cd:
                continue
            cd = cd[0]
        try:
            i = int(cd)
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


_CSV_HEADER = ["index", "filename", "path", "label", "split"]
_EXPORT_HEADER = [*_CSV_HEADER, "sha256", "source", "score", "reason"]


def snapshots_to_csv(snapshots: list[dict]) -> str:
    """Serialize export-list snapshots to CSV (header always emitted).

    Each snapshot carries its manifest sha256 when available, so exported
    lists are content-addressed — traceable across renames and moves.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_EXPORT_HEADER)
    for i, s in enumerate(snapshots):
        score = s.get("score")
        writer.writerow([
            i, s.get("filename", ""), s.get("path", ""),
            s.get("label", ""), s.get("split", ""), s.get("sha256") or "",
            s.get("source", ""), "" if score is None else score,
            s.get("reason", ""),
        ])
    return buf.getvalue()


def records_to_csv(records: list[dict], indices: list[int]) -> str:
    """Serialize selected records to CSV text (header always emitted).

    Columns: index, filename, path, label, split. ``indices`` order is
    preserved.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    for i in indices:
        r = records[i]
        p = Path(r["path"])
        writer.writerow([i, p.name, str(p), r.get("label", ""), r.get("split", "")])
    return buf.getvalue()


def zip_selected_images(records: list[dict], indices: list[int]) -> bytes:
    """Build an in-memory ZIP of the selected image files + a manifest.csv.

    Each image is stored under ``images/<split>/<filename>``; name
    collisions are disambiguated with a numeric suffix. Files missing on
    disk are skipped and recorded with ``status=missing`` in manifest.csv.
    """
    buf = io.BytesIO()
    manifest = io.StringIO()
    mwriter = csv.writer(manifest, lineterminator="\n")
    mwriter.writerow(_CSV_HEADER + ["status", "arcname"])
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in indices:
            r = records[i]
            p = Path(r["path"])
            arcname = f"images/{r.get('split', '')}/{p.name}"
            stem, suffix, n_try = p.stem, p.suffix, 1
            while arcname in used:
                arcname = f"images/{r.get('split', '')}/{stem}_{n_try}{suffix}"
                n_try += 1
            if p.exists():
                zf.write(p, arcname)
                used.add(arcname)
                status = "ok"
            else:
                status, arcname = "missing", ""
            mwriter.writerow(
                [i, p.name, str(p), r.get("label", ""), r.get("split", ""), status, arcname]
            )
        zf.writestr("manifest.csv", manifest.getvalue())
    return buf.getvalue()


def thumbnail_path_for(image_path: Path, size: int = 256) -> Path:
    """Deterministic cache location for an image's thumbnail.

    Lives in ``<image_dir>/.thumbs/<size>/<sha1(abspath|mtime|fsize)>.webp``
    so the key invalidates whenever the source file changes, and same-named
    files in different directories can never collide (per-dir cache).
    Raises OSError if the source file is missing.
    """
    image_path = Path(image_path)
    stat = image_path.stat()
    digest = hashlib.sha1(
        f"{image_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()
    ).hexdigest()[:16]
    return image_path.parent / ".thumbs" / str(size) / f"{digest}.webp"


def make_thumbnail(image_path: Path, size: int = 256) -> Path:
    """Create (or reuse) the cached thumbnail for ``image_path``.

    Returns the thumbnail path. Raises OSError for missing/unreadable
    sources — callers render a placeholder card instead of dropping it.
    """
    image_path = Path(image_path)
    out = thumbnail_path_for(image_path, size)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((size, size))
    img.save(out, "WEBP", quality=80)
    return out


def ensure_thumbnails(
    image_paths: Sequence[Path],
    size: int = 256,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """Pre-generate thumbnails for all paths; returns how many are usable.

    Broken/missing sources are skipped (the grid shows a placeholder for
    them later). ``progress_cb(done, total)`` is called after each path.
    """
    n_total = len(image_paths)
    n_ok = 0
    for i, p in enumerate(image_paths):
        try:
            make_thumbnail(Path(p), size)
            n_ok += 1
        except OSError:
            pass
        if progress_cb is not None:
            progress_cb(i + 1, n_total)
    return n_ok


def spatial_order(
    coords: np.ndarray, indices: Sequence[int], n_rows: int = 12
) -> list[int]:
    """Order ``indices`` to mirror the scatter's spatial layout.

    Top of the plot first (high y), left-to-right (ascending x) within each
    of ``n_rows`` horizontal bands — so the thumbnail grid reads roughly
    like the chart, preserving spatial gestalt without in-plot highlights.
    """
    indices = list(indices)
    if not indices:
        return []
    ys = np.asarray([coords[i, 1] for i in indices], dtype=float)
    xs = np.asarray([coords[i, 0] for i in indices], dtype=float)
    y_min, y_max = float(ys.min()), float(ys.max())
    if y_max == y_min:
        rows = np.zeros(len(indices), dtype=int)
    else:
        rows = ((y_max - ys) / (y_max - y_min) * (n_rows - 1e-9)).astype(int)
    order = sorted(range(len(indices)), key=lambda j: (int(rows[j]), float(xs[j])))
    return [indices[j] for j in order]


def yolo_label_path_for(image_path: Path) -> Path:
    """Map an image path under ``<root>/images/x.jpg`` to its YOLO label
    file ``<root>/labels/x.txt`` (detector-mode dataset layout)."""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def draw_yolo_boxes(
    image_path: Path,
    label_path: Path,
    class_names: list[str] | None = None,
) -> Image.Image:
    """Return the image with its YOLO boxes drawn (red, 2px, class tag).

    A missing/empty label file yields the unmodified image. Lines that fail
    to parse are skipped.
    """
    img = Image.open(image_path).convert("RGB")
    label_path = Path(label_path)
    if not label_path.exists():
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline="#e74c3c", width=2)
        name = (
            class_names[cid]
            if class_names and 0 <= cid < len(class_names)
            else f"class_{cid}"
        )
        draw.text((x0 + 2, max(0, y0 - 12)), name, fill="#e74c3c")
    return img


# ── embedding-space coverage / gap-filling (defect-mechanisms 嵌入覆蓋圖) ──
# The honest counterpart to the attribute-axis completeness grid: density is
# measured HERE, in raw cosine embedding space (k-NN distance), and a 2-D
# projection is only ever used to *draw* the regions found in high-D — never
# to count them (which is what completeness.py's docstring rightly refuses).

def sparsity_scores(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-row data sparsity in raw embedding space: mean cosine distance to
    each row's k nearest OTHER rows. High = the model was shown few similar
    examples near here → a coverage blind-spot candidate.

    A thin, self-referential wrapper over compute_outlier_scores (drop-self),
    so the manifold scatter can be coloured by sparsity computed in full
    dimensionality, not off the distorted 2-D layout. <2 rows → all zeros.
    """
    emb = np.asarray(embeddings)
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=float)
    return compute_outlier_scores(emb, emb, k=k, candidates_in_reference=True)


def rank_gap_fillers(
    candidate_embeddings: np.ndarray,
    dataset_embeddings: np.ndarray,
    k: int = 10,
    top: int | None = None,
    min_distance: float | None = None,
) -> tuple[list[int], list[float]]:
    """Rank a NEW folder's images by how much each fills a SPARSE region of
    the existing dataset, in raw cosine space (the new-folder gap-filling
    score for the 嵌入覆蓋圖 view).

    Score = mean cosine distance from a candidate to its k nearest DATASET
    neighbours (large = lands where the dataset is thin = better gap-filler).
    Returns (candidate indices, scores) sorted DESCENDING. ``min_distance``
    drops candidates closer than the threshold (already covered, not a gap);
    ``top`` caps the list. Honest by construction: a high score means "your
    data is thin here", NOT "the model is proven weak here" — the H1–H5 gate
    (diagnose_sparse_points) decides whether collecting actually helps.
    """
    cand = np.asarray(candidate_embeddings)
    data = np.asarray(dataset_embeddings)
    if len(cand) == 0 or len(data) == 0:
        return [], []
    scores = compute_outlier_scores(cand, data, k=k)
    order = np.argsort(scores)[::-1]
    out_idx: list[int] = []
    out_score: list[float] = []
    for i in order:
        s = float(scores[i])
        if min_distance is not None and s < min_distance:
            continue
        out_idx.append(int(i))
        out_score.append(s)
    if top is not None:
        out_idx, out_score = out_idx[:top], out_score[:top]
    return out_idx, out_score


def nearest_labels(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_labels: Sequence[str],
) -> list[str]:
    """1-NN label transfer: each query row gets its nearest reference row's
    label (cosine). Seeds PROVISIONAL labels for unlabeled gap-filling
    candidates so the blind quiz has a class to render — never a ground-truth
    claim. Empty reference (or labels) → "" for every query.
    """
    q = np.asarray(query_embeddings)
    ref = np.asarray(reference_embeddings)
    labels = list(reference_labels)
    if len(q) == 0:
        return []
    if len(ref) == 0 or not labels:
        return [""] * len(q)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(ref)
    _, idx = nn.kneighbors(q)
    return [labels[int(row[0])] for row in idx]


def candidates_to_quiz_records(
    candidate_records: Sequence[dict],
    provisional_labels: Sequence[str],
    gap_scores: Sequence[float] | None = None,
) -> list[dict]:
    """Wrap gap-filling candidates as quiz-ready records (the send-to-quiz
    handoff contract).

    Each output carries a PROVISIONAL ``label`` (from nearest_labels) so the
    blind quiz can render answer buttons, plus provenance — ``source``,
    ``provisional=True``, ``gap_score`` — so the quiz UI can flag these as
    unlabeled candidates rather than graded golden cases. ``split`` defaults
    to "candidate".
    """
    out: list[dict] = []
    for i, r in enumerate(candidate_records):
        score = (float(gap_scores[i])
                 if gap_scores is not None and i < len(gap_scores) else None)
        out.append({
            "path": r["path"],
            "split": r.get("split", "candidate"),
            "label": provisional_labels[i] if i < len(provisional_labels) else "",
            "source": "gap_filler",
            "provisional": True,
            "gap_score": score,
        })
    return out


def diagnose_sparse_points(
    embeddings: np.ndarray,
    labels: Sequence[str],
    indices: Sequence[int],
    *,
    s1_consistency: float,
    radius: float,
    k: int = 20,
    signal_level_for: Callable[[int], str] | None = None,
) -> list[dict]:
    """Run the H0/H1–H5 root-cause gate over a set of sparse-region points so
    the coverage map never bare-claims "sparse → go collect".

    Per point, three orthogonal signals feed diagnose_root_cause:
      S2 = neighbor_hit_density(radius) — this point's local data coverage,
      S3 = neighbor_label_entropy(k)    — proxy model uncertainty,
      S1 = s1_consistency               — human consistency (supplied; from
           the quiz / gauge-R&R, shared across the region).
    ``signal_level_for`` (global index → 桶① signal level, e.g. from
    signal_strength) adds the S0 physical-detectability screen: a point whose
    signal is 確無 is reported as H0 物理天花板 (collecting cannot help) rather
    than a coverage gap. Without it the gate runs exactly as before.

    Returns [{idx, cause, add_data, s2_density, s3_entropy, signal_level,
    caveat}] so the UI can report how many sparse points are H1/H5
    (collecting helps), H0 (physically unrecoverable) vs H2/H3/H4.
    """
    out: list[dict] = []
    for i in indices:
        i = int(i)
        dens = neighbor_hit_density(embeddings, i, radius)
        ent = neighbor_label_entropy(embeddings, labels, i, k=k)
        sig = signal_level_for(i) if signal_level_for is not None else None
        diag = diagnose_root_cause(s1_consistency, dens, ent, signal_level=sig)
        out.append({
            "idx": i, "cause": diag["cause"], "add_data": diag["add_data"],
            "s2_density": dens, "s3_entropy": round(float(ent), 3),
            "signal_level": diag["signal_level"], "caveat": diag["caveat"],
        })
    return out


# ── object-level coverage (crop each YOLO bbox → one point per OBJECT) ─────
# The whole-image embedding collapses a multi-object detection scene into a
# single point, so "sparse" means "unusual scene", not "unusual object". For
# detection data the actionable unit is the object: crop each bbox, embed the
# crop, and measure sparsity per object. These helpers stay pure (geometry +
# label parsing); the app does the disk crop/cache around them.

def parse_yolo_boxes(
    label_path: Path,
) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file → ``[(class_id, cx, cy, w, h), …]`` with
    normalized (0–1) box coords. Missing file, short lines, non-numeric
    fields, and non-positive boxes are skipped (never raises)."""
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    out: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        out.append((cid, cx, cy, w, h))
    return out


def bbox_to_pixels(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int, pad: float = 0.0,
) -> tuple[int, int, int, int]:
    """Normalized YOLO bbox → integer pixel box ``(x0, y0, x1, y1)``, grown by
    ``pad`` (fraction of each side) for context and clamped to the image. The
    returned box always has positive area (≥1px) even for degenerate input."""
    gw, gh = w * (1.0 + 2.0 * pad), h * (1.0 + 2.0 * pad)
    x0 = max(0, int(round((cx - gw / 2.0) * img_w)))
    y0 = max(0, int(round((cy - gh / 2.0) * img_h)))
    x1 = min(img_w, int(round((cx + gw / 2.0) * img_w)))
    y1 = min(img_h, int(round((cy + gh / 2.0) * img_h)))
    if x1 <= x0:
        x1 = min(img_w, x0 + 1)
    if y1 <= y0:
        y1 = min(img_h, y0 + 1)
    return x0, y0, x1, y1


def crop_bbox(
    img: Image.Image, cx: float, cy: float, w: float, h: float, pad: float = 0.0,
) -> Image.Image:
    """Crop a normalized YOLO bbox out of a PIL image (padded, clamped)."""
    iw, ih = img.size
    box = bbox_to_pixels(cx, cy, w, h, iw, ih, pad=pad)
    return img.crop(box)


def discover_yolo_objects(
    image_paths: Sequence[Path],
    class_names: Sequence[str] | None = None,
    label_for: Callable[[Path], Path] | None = None,
) -> list[dict]:
    """Expand detection images into one record PER OBJECT by reading each
    image's YOLO label file.

    Returns ``[{image_path, label, class_id, bbox=(cx,cy,w,h), obj_index}, …]``
    in image-then-box order. Images with no label file / no boxes contribute
    nothing. ``label_for`` resolves an image path to its label file (default
    :func:`yolo_label_path_for`); ``class_names`` maps class_id → name
    (fallback ``"class_<id>"``).
    """
    resolve = label_for or yolo_label_path_for
    names = list(class_names) if class_names else None
    out: list[dict] = []
    for ip in image_paths:
        ip = Path(ip)
        for k, (cid, cx, cy, w, h) in enumerate(parse_yolo_boxes(resolve(ip))):
            label = (names[cid] if names and 0 <= cid < len(names)
                     else f"class_{cid}")
            out.append({
                "image_path": ip, "label": label, "class_id": cid,
                "bbox": (cx, cy, w, h), "obj_index": k,
            })
    return out


def cross_class_nn_pairs(
    embeddings: np.ndarray,
    labels: Sequence[str],
    k: int = 1,
    max_pairs: int = 200,
) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` where j is among i's k nearest neighbours in cosine
    space but carries a DIFFERENT label — the "close but different class"
    conflicts the eye catches on the embedding plot (the label-disagreement
    signal, drawn as connecting lines).

    Deduped as unordered pairs and capped at ``max_pairs``, closest first
    (the most blatant conflicts). <2 rows → empty.
    """
    emb = np.asarray(embeddings)
    labels = list(labels)
    n = len(emb)
    if n < 2:
        return []
    kk = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)
    seen: set[tuple[int, int]] = set()
    cand: list[tuple[int, int, float]] = []
    for i in range(n):
        for col in range(1, kk + 1):       # col 0 is the point itself
            j = int(idx[i, col])
            if labels[j] == labels[i]:
                continue
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            cand.append((key[0], key[1], float(dist[i, col])))
    cand.sort(key=lambda t: t[2])
    return [(i, j) for i, j, _ in cand[:max_pairs]]


def reference_coverage(
    emb_a: np.ndarray, emb_b: np.ndarray, radius: float,
) -> tuple[list[int], float, np.ndarray]:
    """A 相對『外部參照分佈 B』的覆蓋（嵌入覆蓋圖的關係②，與自我參照稀疏互補）。

    對每個 B 點算到 A 的最近 cosine 距離 ``d_b_to_a``；超過 ``radius`` 視為
    『A 沒覆蓋到的 B 區域』。回傳 ``(uncovered_b_indices, recall, d_b_to_a)``，
    其中 ``recall`` = 落在半徑內的 B 比例（A 覆蓋了參照的多少；1.0＝全覆蓋）。
    uncovered 依距離由遠到近排序（最該補的在前）。空輸入 → ([], 1.0, 空陣列)。

    這是覆蓋圖第一次有「外部真值」：自我參照稀疏看不出「全資料集都缺的類型」，
    而以 B 當參照就能誠實量出『相對 B 我缺哪裡』。
    """
    a = np.asarray(emb_a)
    b = np.asarray(emb_b)
    if len(a) == 0 or len(b) == 0:
        return [], 1.0, np.zeros(len(b), dtype=float)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(a)
    dist, _ = nn.kneighbors(b)
    d_b_to_a = dist[:, 0]
    uncovered = [int(i) for i in np.argsort(d_b_to_a)[::-1] if d_b_to_a[i] > radius]
    recall = 1.0 - len(uncovered) / len(b)
    return uncovered, float(recall), d_b_to_a
