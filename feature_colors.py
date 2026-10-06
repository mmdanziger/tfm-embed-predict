"""
feature_colors.py
-----------------
        
Central color map for all benchmark plots.

Scheme
------
  Blue   → simple linear baselines  : PCA-50, Raw Log-norm
  Neutral → random / chance baseline : random_proj50
  Orange/Red/Magenta → foundation models : everything else
"""

# ── Palette definitions ─────────────────────────────────────────────────────

# Blues (baselines)
_BLUE_DARK   = "#1f6eb5"   # PCA-50
_BLUE_LIGHT  = "#6baed6"   # Raw Log-norm

# Neutral (random)
_NEUTRAL     = "#9e9e9e"   # random_proj50

# Foundation-model palette (orange → red → magenta cycle)
"""
_FM_COLORS = [
    "#b2182b",   # red          – BMFM_LW
    "#d6604d",   # orange       – BMFM_CONCAT
    "#f4a582",   # yellow       – BMFM_LM
    "#fddbc7",   # pink         – BMFM_BW
    "#1b9e77",   # brown shade1 – GENEFORMER
    "#e66101",   # brown shade2 – SCVI
    "#998ec3",   # light purple – TF-EXEMPLAR-HUMAN
    "#542788",   # deep purple  – TF-SAPIENS
]
"""
_FM_COLORS = [
    "#006d2c",   # red          – BMFM_LW
    "#31a354",   # orange       – BMFM_CONCAT
    "#74c476",   # yellow       – BMFM_LM
    "#bae4b3",   # pink         – BMFM_BW
    "#b2182b",   # brown shade1 – GENEFORMER
    "#e66101",   # brown shade2 – SCVI
    "#998ec3",   # light purple – TF-EXEMPLAR-HUMAN
    "#542788",   # deep purple  – TF-SAPIENS
]

# ── Master map ───────────────────────────────────────────────────────────────
# Keys are the `feature_clean` strings produced by load_and_prepare_results().
# Add / rename entries here and every plot picks up the change automatically.

FEATURE_COLOR_MAP: dict[str, str] = {
    # ── Baselines (blue) ──────────────────────────────────────────────────
    "PCA-50":        _BLUE_DARK,
    "Raw Log-norm":  _BLUE_LIGHT,
    # ── Random / neutral ─────────────────────────────────────────────────
    "random_proj50": _NEUTRAL,
    # ── Foundation models (orange / red / magenta) ────────────────────────
    "BMFM_LW":               _FM_COLORS[0],
    "BMFM_CONCAT":          _FM_COLORS[1],
    "GENEFORMER":         _FM_COLORS[4],
    "SCVI":               _FM_COLORS[5],
    "TF-EXEMPLAR-HUMAN":  _FM_COLORS[6],
    "TF-SAPIENS":         _FM_COLORS[7],
    "BMFM_LM":        _FM_COLORS[2],   # ← new
    "BMFM_BW":        _FM_COLORS[3],
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_palette(features: list[str]) -> list[str]:
    """
    Return a list of colors aligned with `features`.

    Unknown features fall back to the next unused FM color, then gray.

    Example
    -------
    >>> palette = get_palette(feature_order)   # feature_order from your df
    >>> sns.boxplot(..., palette=palette)
    """
    used_fm = iter(_FM_COLORS)
    result = []
    for f in features:
        if f in FEATURE_COLOR_MAP:
            result.append(FEATURE_COLOR_MAP[f])
        else:
            # Auto-assign next FM color for unknown embeddings
            result.append(next(used_fm, "#555555"))
    return result


def get_color(feature: str) -> str:
    """Return the single color for one feature name."""
    return FEATURE_COLOR_MAP.get(feature, "#555555")