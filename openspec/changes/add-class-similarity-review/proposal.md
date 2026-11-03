## Why

Visualize already shows class-confusion heatmaps and "neighbors mostly another class" candidates, but those scores are not the same quantity as per-record same-class versus nearest-rival Top-K cosine. Reviewers need a distribution they can drill into, a margin threshold that previews workload, and Precision/Recall only after a frozen human verdict that the original label is wrong.

## What Changes

- Extend the existing Visualize「標錯偵測／類別混淆」panel with「類別相似度分布」.
- Add pure functions for blocked exact Top-K margin, record identity, threshold preview, human evaluation, and schema-2 export.
- Wire selection/cart/download through existing `viz_selection` and `_add_to_export`.
- Do not add a top-level tool, revive gray-band review, auto-relabel, or change Labeling contracts.

## Capabilities

### New Capabilities
- `class-similarity-review`: class-wise same/other/margin distribution, threshold candidate preview, human-evaluation Precision/Recall, and reproducible evidence export.

### Modified Capabilities
- Visualize label-quality panel: add a dispatched view beside existing confusion/mismatch tools; invalidate on readback and generation change.

## Impact

New modules under `scripts/class_similarity*.py`, tests, usage-guide section, OpenSpec/U-Net contracts. Reuses `viz_raw_embeddings`, records, cart, and the Playwright harness. No new remote service. Source datasets remain read-only. Existing dirty workspace changes are left untouched.
