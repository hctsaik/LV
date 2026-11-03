# Grad-CAM analysis resolution

The Visualize Grad-CAM panel exposes three native DINOv2 input resolutions:

| Input | Native patch grid | Use |
|---:|---:|---|
| 224 | 16×16 | Fast, and the default classifier-head recipe |
| 336 | 24×24 | Exact session head with a finer native grid |
| 448 | 32×32 | Exact session head with the finest supported grid |

DINOv2 keeps a 14-pixel patch size. Selecting 336 or 448 therefore runs the
model at a genuinely denser patch grid; it does not interpolate a 16×16 map.

The default 224 head is not silently reused at higher resolution. When a user
selects 336 or 448 and the exact session head is absent, Visualize performs a
foreground full-dataset embedding pass at that resolution and trains only the
frozen-backbone classifier head. The head registry key includes model,
pooling, scope, crop/aspect policy, and resolution. If the full session dataset
is unavailable (for example in an inspector-only context), the mismatched
resolution is disabled rather than being presented as an exact head.

For `meanpool`, token-wise gradient×activation is computed on the normalized
tokens that actually feed the meanpool embedding. Computing it on the
pre-normalization tokens would collapse to an almost-zero attribution because
the L2 normalization is scale-invariant.

The resulting map is still classification evidence, not pixel-accurate crack
segmentation. Thin defects may require the 336/448 probe or a dedicated
patch-level anomaly/segmentation model.
