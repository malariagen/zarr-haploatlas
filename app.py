import time
import pandas as pd
import numpy as np
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, resolve_loci, build_regions,
    build_variant_rows, load_call_data, build_allele_matrix,
)
from src.haplotypes import deduplicate_allele_matrix, compute_haplotypes

st.set_page_config(layout="wide", page_title="Variant Marketplace", page_icon = "assets/logo.svg")

# ── Load static data ──────────────────────────────────────────────────────────
chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── User input ────────────────────────────────────────────────────────────────

st.logo("assets/logo.svg", size = "large")
st.title("Variant Marketplace", text_alignment="center")

DEBUG = st.toggle("Debug mode", value=False)

RAW_USER_INPUT = st.text_area(
    "Enter genomic loci",
    value="PF3D7_0709000[72-76,220,271] Pf3D7_04_v3[104205,139150-139156]",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Multiple loci separated by spaces."
    ),
)

parsed_loci   = parse_loci_from_input(RAW_USER_INPUT)
resolved_loci = resolve_loci(parsed_loci, reference_files["cds_gff"])

if parsed_loci.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()


if DEBUG:
    with st.expander("Debug"):
        st.write("**Chunk index** (all chromosomes in zarr)")
        st.dataframe(chunk_index_df, width="stretch", hide_index=True)

        st.write("**Parsed loci:**")
        st.dataframe(parsed_loci, width="stretch", hide_index=True)

        st.write("**Resolved genomic intervals:**")
        for sid, info in resolved_loci.items():
            st.text(f"{sid} ({info['coord_type']}): {info['intervals']}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Variant metadata table
# ══════════════════════════════════════════════════════════════════════════════

regions, warnings = build_regions(resolved_loci, variant_data, chunk_index_df)
for w in warnings:
    st.warning(w)

if not regions:
    st.warning("No variants found for any queried locus.")
    st.stop()

# ── Variant metadata table ────────────────────────────────────────────────────
st.divider()
st.subheader("Overview of variants of interest")

meta_rows = []
for source_id, region in regions.items():
    meta_rows.extend(build_variant_rows(source_id, region["meta"], resolved_loci[source_id]))

meta_df    = pd.DataFrame(meta_rows)
n_samples  = next(iter(regions.values()))["meta"]["n_samples"]
total_vars = sum(r["meta"]["n_variants"] for r in regions.values())

# ── Filter toggles ────────────────────────────────────────────────────────────
fcol1, fcol2, _ = st.columns(3)
apply_filter_pass = fcol1.toggle(
    "Filter pass variants only",
    value=False,
    help="Exclude variants that do not pass quality filters",
)
apply_numalt1 = fcol2.toggle(
    "Biallelic variants only",
    value=False,
    help="Exclude multi-allelic variant sites",
)

fail_fp  = ~meta_df["filter_pass"] if apply_filter_pass else pd.Series(False, index=meta_df.index)
fail_na  = (meta_df["numalt"] != 1) if apply_numalt1    else pd.Series(False, index=meta_df.index)
fail_any = fail_fp | fail_na

excluded_positions: set[str] = {
    str(meta_df.loc[i, "position"]) for i in meta_df.index if fail_any.loc[i]
}

# ── Styled display ────────────────────────────────────────────────────────────
_BOOL_COLS = ["is_snp", "filter_pass", "CDS", "needs_translation"]

def _highlight_failed(row):
    color = "background-color: #ffcccc" if fail_any.loc[row.name] else ""
    return [color] * len(row)

def _color_bool(val):
    if val == "✓":
        return "color: #2d8a4e"
    if val == "✗":
        return "color: #cc3333"
    return ""

display_meta = meta_df.copy()
for col in _BOOL_COLS:
    if col in display_meta.columns:
        display_meta[col] = display_meta[col].map({True: "✓", False: "✗"})

styled_meta = (
    display_meta.style
    .apply(_highlight_failed, axis=1)
    .map(_color_bool, subset=_BOOL_COLS)
)
st.dataframe(
    styled_meta,
    hide_index=True,
    width="stretch",
    column_config={
        "alt":    st.column_config.ListColumn("alt alleles"),
        "numalt": None,
    },
)
st.caption(
    f"{total_vars} variant sites across {len(regions)} {'locus' if len(regions) == 1 else 'loci'}, "
    f"affecting {n_samples:,} samples."
    + (" Rows highlighted in red are excluded from downstream steps." if fail_any.any() else ""),
    text_alignment="right",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Haplotypes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Build haplotypes")

_HET_LABELS = {
    "Exclude and only use hom calls": "exclude",
    "Use the major allele":           "major_ad",
    "Order alleles by depth":         "ordered_ad",
}
het_mode_label = st.radio(
    "How to handle heterozygous genotypes when computing haplotypes:",
    list(_HET_LABELS.keys()),
    horizontal=True,
)
HET_MODE = _HET_LABELS[het_mode_label]

# Small state management section
_load_state = (RAW_USER_INPUT, apply_filter_pass, apply_numalt1, HET_MODE)
if st.session_state.get("last_load_state") != _load_state:
    st.session_state["last_load_state"]   = _load_state
    st.session_state["haplotypes_built"] = False

if not st.session_state.get("haplotypes_built"):
    if st.button("Build haplotypes", type="primary"):
        st.session_state["haplotypes_built"] = True
        st.rerun()
else:
    # Read the save-intermediates toggle value from the previous run so the
    # computation can decide whether to stash allele_matrix before freeing it.
    # (The toggle itself is defined later in the debug expander; session state
    # bridges the forward reference across reruns.)
    _save_intermediates = st.session_state.get("_debug_save_intermediates", False)

    t0 = time.time()
    # Process one region at a time to cap peak memory — loading all genotype arrays
    # simultaneously (n_loci × n_variants × n_samples) can exceed the Cloud limit.
    partial_matrices = []
    region_ids  = list(regions.keys())
    n_regions   = len(region_ids)
    progress    = st.progress(0, text="Loading genotypes…")
    for i, source_id in enumerate(region_ids):
        region = regions[source_id]
        progress.progress(i / n_regions, text=f"Loading {source_id} ({i + 1}/{n_regions})…")
        region["genotypes"], region["g1_wins"] = load_call_data(
            region["ds"],
            cache_key=(source_id, apply_filter_pass, apply_numalt1),
            load_ad=(HET_MODE in ("major_ad", "ordered_ad")),
        )
        partial_matrices.append(
            build_allele_matrix({source_id: region}, excluded_positions, HET_MODE)
        )
        region["genotypes"] = None
        region["g1_wins"]   = None
    progress.empty()

    allele_matrix = pd.concat(partial_matrices, axis=1)
    partial_matrices.clear()

    if _save_intermediates:
        st.session_state["_debug_allele_matrix"] = allele_matrix.copy()

    deduped = deduplicate_allele_matrix(allele_matrix)
    allele_matrix = None  # free before haplotype computation

    raw = compute_haplotypes(
        deduped, regions, resolved_loci, parsed_loci, reference_files["cds_gff"]
    )

    st.success(f"Loaded in {time.time() - t0:.1f}s")

    if DEBUG:
        with st.expander("Debug"):
            st.toggle(
                "Save intermediate outputs",
                key="_debug_save_intermediates",
                help="When on, the allele matrix is preserved in session state before being freed. "
                     "Flip on, then re-run by changing any input.",
            )
            if _save_intermediates and "_debug_allele_matrix" in st.session_state:
                st.write("**Allele matrix** (samples × positions)")
                st.dataframe(st.session_state["_debug_allele_matrix"], hide_index=False, width="stretch")

            st.write("**Deduplicated allele matrix**")
            st.dataframe(deduped, hide_index=True, width="stretch")

            st.write("**Haplotype output**")
            st.dataframe(raw, hide_index=True, width="stretch")


# if not genotypes_loaded:
#     st.info(f"Click below to retrieve haplotypes for {n_samples:,} samples across all loci.")
#     if st.button("Retrieve haplotypes", type="primary"):
#         t0 = time.time()
#         with st.spinner(f"Fetching genotypes for {n_samples:,} samples…"):
#             for region in regions.values():
#                 region["genotypes"]     = load_genotypes(region["ds"])
#                 region["allele_depths"] = load_allele_depths(region["ds"])
#         st.toast(f"✅ Loaded in {time.time() - t0:.1f}s")
#         st.rerun()
# else:
#     allele_matrix = build_allele_matrix(
#         regions,
#         excluded_positions=excluded_positions,
#         het_mode=HET_MODE,
#     )

#     if st.button("Compute haplotypes", type="primary"):
#         t0 = time.time()
#         with st.spinner("Computing haplotypes…"):
#             deduped = deduplicate_allele_matrix(allele_matrix)
#             raw = compute_haplotypes(
#                 deduped, regions, resolved, loci_df, reference_files["cds_gff"]
#             )

#             raw = raw.sort_values("n_samples", ascending=False).reset_index(drop=True)
#             w   = int(np.log10(max(len(raw), 1))) + 1
#             raw.insert(0, "combination_id", [f"C{i+1:0{w}d}" for i in range(len(raw))])

#             st.session_state["sample_to_id"] = {
#                 sid: row["combination_id"]
#                 for _, row in raw.iterrows()
#                 for sid in row["sample_ids"]
#             }

#             hap_cols = [c for c in raw.columns if c.endswith(("_haplotype", "_ns_changes"))]
#             # Per-range columns (named, non-numeric) to carry through the groupby
#             _skip = {"combination_id", "n_samples", "sample_ids"} | set(hap_cols)
#             per_pos_from_raw = [
#                 c for c in raw.columns
#                 if c not in _skip and not c.lstrip("-").isdigit()
#             ]
#             agg_spec: dict = {"n_samples": ("n_samples", "sum"), "combination_ids": ("combination_id", list)}
#             for c in per_pos_from_raw:
#                 agg_spec[c] = (c, "first")
#             haplotypes = (
#                 raw.groupby(hap_cols, sort=False)
#                 .agg(**agg_spec)
#                 .reset_index()
#                 .sort_values("n_samples", ascending=False)
#                 .reset_index(drop=True)
#             )
#             wh = int(np.log10(max(len(haplotypes), 1))) + 1
#             haplotypes.insert(0, "haplotype_id", [f"H{i+1:0{wh}d}" for i in range(len(haplotypes))])

#             st.session_state["haplotype_result"] = haplotypes
#             st.session_state["haplotype_raw"]    = raw
#         st.toast(f"✅ Computed in {time.time() - t0:.2f}s")
#         st.rerun()

#     if "haplotype_result" in st.session_state:
#         result   = st.session_state["haplotype_result"]
#         raw      = st.session_state["haplotype_raw"]

#         hap_cols = [c for c in result.columns if c.endswith(("_haplotype", "_ns_changes"))]
#         fixed_cols = {"haplotype_id", "n_samples", "combination_ids"} | set(hap_cols)
#         per_pos_cols = [
#             c for c in result.columns
#             if c not in fixed_cols and not c.lstrip("-").isdigit()
#         ]

#         # ── Haplotype summary ──────────────────────────────────────────────
#         st.subheader("Haplotype summary")
#         summary_cols = ["haplotype_id", "n_samples", "combination_ids"] + hap_cols + per_pos_cols
#         st.dataframe(
#             result[[c for c in summary_cols if c in result.columns]],
#             width="stretch", hide_index=True,
#         )

#         # ── Per-sample haplotypes ──────────────────────────────────────────
#         _fixed_raw = {"combination_id", "n_samples", "sample_ids"}
#         _hap_raw   = {c for c in raw.columns if c.endswith(("_haplotype", "_ns_changes"))}
#         # Exclude raw allele-matrix position columns (pure numeric strings like "139150");
#         # keep only the named per-range columns added by compute_haplotypes.
#         per_pos_raw = [
#             c for c in raw.columns
#             if c not in _fixed_raw and c not in _hap_raw and not c.lstrip("-").isdigit()
#         ]

#         if per_pos_raw:
#             per_sample = (
#                 raw[["combination_id", "sample_ids"] + per_pos_raw]
#                 .explode("sample_ids")
#                 .rename(columns={"sample_ids": "sample_id"})
#                 .reset_index(drop=True)
#             )
#             st.subheader("Per-sample haplotypes")
#             st.dataframe(per_sample, width="stretch", hide_index=True)
