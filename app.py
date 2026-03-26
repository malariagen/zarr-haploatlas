import time
import pandas as pd
import numpy as np
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, resolve_loci,
    query_locus_metadata, filter_region_by_intervals,
    build_variant_rows, load_genotypes, load_allele_depths, build_allele_matrix,
)
from src.haplotypes import deduplicate_allele_matrix, compute_haplotypes

st.set_page_config(layout="wide", page_title="FIX ME")

# ── Load static data ──────────────────────────────────────────────────────────
chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── User input ────────────────────────────────────────────────────────────────
RAW_USER_INPUT = st.text_input(
    "Enter genomic loci",
    value="PF3D7_0709000[72-76,220,271] Pf3D7_04_v3[104205,139150-139156]",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Multiple loci separated by spaces."
    ),
)

if st.session_state.get("last_input") != RAW_USER_INPUT:
    st.session_state.pop("regions", None)
    st.session_state.pop("haplotype_result", None)
    st.session_state.pop("haplotype_raw", None)
    st.session_state.pop("sample_to_id", None)
    st.session_state["last_input"] = RAW_USER_INPUT

loci_df  = parse_loci_from_input(RAW_USER_INPUT)
resolved = resolve_loci(loci_df, reference_files["cds_gff"])

if loci_df.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Variant metadata table
# ══════════════════════════════════════════════════════════════════════════════

UPSTREAM_PAD = 100  # bp

if "regions" not in st.session_state:
    regions = {}
    for source_id, locus_info in resolved.items():
        intervals = locus_info["intervals"]
        if not intervals:
            st.warning(f"No genomic intervals resolved for `{source_id}`")
            continue

        padded = tuple((c, max(1, s - UPSTREAM_PAD), e) for c, s, e in intervals)
        result = query_locus_metadata(variant_data, chunk_index_df, padded)
        if result is None:
            st.warning(f"No variants found for `{source_id}`")
            continue

        filtered = filter_region_by_intervals(*result, intervals)
        if filtered is None:
            st.warning(f"No variants found for `{source_id}`")
            continue

        meta, ds = filtered
        regions[source_id] = {"meta": meta, "ds": ds, "genotypes": None, "allele_depths": None}

    st.session_state["regions"] = regions

regions = st.session_state["regions"]

if not regions:
    st.warning("No variants found for any queried locus.")
    st.stop()

st.header("Variant Sites")
meta_rows = []
for source_id, region in regions.items():
    meta_rows.extend(build_variant_rows(source_id, region["meta"], resolved[source_id]))

meta_df    = pd.DataFrame(meta_rows)
n_samples  = next(iter(regions.values()))["meta"]["n_samples"]
total_vars = sum(r["meta"]["n_variants"] for r in regions.values())

# ── Filter toggles ────────────────────────────────────────────────────────────
fcol1, fcol2 = st.columns(2)
with fcol1:
    apply_filter_pass = st.toggle(
        "Filter: PASS only",
        value=False,
        help="Highlight and exclude variants that do not pass quality filters",
    )
with fcol2:
    apply_numalt1 = st.toggle(
        "Filter: numalt = 1 only",
        value=False,
        help="Highlight and exclude multi-allelic variant sites",
    )

# Rows that fail the active filters
fail_fp  = ~meta_df["filter_pass"] if apply_filter_pass else pd.Series(False, index=meta_df.index)
fail_na  = (meta_df["numalt"] != 1) if apply_numalt1  else pd.Series(False, index=meta_df.index)
fail_any = fail_fp | fail_na

excluded_positions: set[str] = {
    str(meta_df.loc[i, "position"]) for i in meta_df.index if fail_any.loc[i]
}


def _highlight_failed(row):
    color = "background-color: #ffcccc" if fail_any.loc[row.name] else ""
    return [color] * len(row)


styled_meta = meta_df.style.apply(_highlight_failed, axis=1)
st.dataframe(styled_meta, hide_index=True)
st.caption(
    f"{total_vars} variant sites across {len(regions)} loci · "
    f"{n_samples:,} samples · "
    f"Queried with {UPSTREAM_PAD}bp upstream padding to capture "
    f"variants with long REF alleles spanning into the region."
)

# ── Invalidate haplotype cache when filter/het state changes ──────────────────
filter_state = (apply_filter_pass, apply_numalt1)
if st.session_state.get("last_filter_state") != filter_state:
    st.session_state.pop("haplotype_result", None)
    st.session_state.pop("haplotype_raw", None)
    st.session_state.pop("sample_to_id", None)
    st.session_state["last_filter_state"] = filter_state


# ══════════════════════════════════════════════════════════════════════════════
# 2. Haplotypes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Haplotypes")

_HET_LABELS = {
    "Mark as * (het)":          "star",
    "Major allele by depth":    "major_ad",
    "All alleles by depth":     "all_ad",
}
het_mode_label = st.radio(
    "Heterozygous call handling",
    list(_HET_LABELS.keys()),
    horizontal=True,
)
HET_MODE = _HET_LABELS[het_mode_label]

if st.session_state.get("last_het_mode") != HET_MODE:
    st.session_state.pop("haplotype_result", None)
    st.session_state.pop("haplotype_raw", None)
    st.session_state.pop("sample_to_id", None)
    st.session_state["last_het_mode"] = HET_MODE

genotypes_loaded = all(r["genotypes"] is not None for r in regions.values())

if not genotypes_loaded:
    st.info(f"Click below to retrieve haplotypes for {n_samples:,} samples across all loci.")
    if st.button("Retrieve haplotypes", type="primary"):
        t0 = time.time()
        with st.spinner(f"Fetching genotypes for {n_samples:,} samples…"):
            for region in regions.values():
                region["genotypes"]     = load_genotypes(region["ds"])
                region["allele_depths"] = load_allele_depths(region["ds"])
        st.toast(f"✅ Loaded in {time.time() - t0:.1f}s")
        st.rerun()
else:
    allele_matrix = build_allele_matrix(
        regions,
        excluded_positions=excluded_positions,
        het_mode=HET_MODE,
    )

    if st.button("Compute haplotypes", type="primary"):
        t0 = time.time()
        with st.spinner("Computing haplotypes…"):
            deduped = deduplicate_allele_matrix(allele_matrix)
            raw = compute_haplotypes(
                deduped, regions, resolved, loci_df, reference_files["cds_gff"]
            )

            raw = raw.sort_values("n_samples", ascending=False).reset_index(drop=True)
            w   = int(np.log10(max(len(raw), 1))) + 1
            raw.insert(0, "combination_id", [f"C{i+1:0{w}d}" for i in range(len(raw))])

            st.session_state["sample_to_id"] = {
                sid: row["combination_id"]
                for _, row in raw.iterrows()
                for sid in row["sample_ids"]
            }

            hap_cols = [c for c in raw.columns if c.endswith(("_haplotype", "_ns_changes"))]
            # Per-range columns (named, non-numeric) to carry through the groupby
            _skip = {"combination_id", "n_samples", "sample_ids"} | set(hap_cols)
            per_pos_from_raw = [
                c for c in raw.columns
                if c not in _skip and not c.lstrip("-").isdigit()
            ]
            agg_spec: dict = {"n_samples": ("n_samples", "sum"), "combination_ids": ("combination_id", list)}
            for c in per_pos_from_raw:
                agg_spec[c] = (c, "first")
            haplotypes = (
                raw.groupby(hap_cols, sort=False)
                .agg(**agg_spec)
                .reset_index()
                .sort_values("n_samples", ascending=False)
                .reset_index(drop=True)
            )
            wh = int(np.log10(max(len(haplotypes), 1))) + 1
            haplotypes.insert(0, "haplotype_id", [f"H{i+1:0{wh}d}" for i in range(len(haplotypes))])

            st.session_state["haplotype_result"] = haplotypes
            st.session_state["haplotype_raw"]    = raw
        st.toast(f"✅ Computed in {time.time() - t0:.2f}s")
        st.rerun()

    if "haplotype_result" in st.session_state:
        result   = st.session_state["haplotype_result"]
        raw      = st.session_state["haplotype_raw"]

        hap_cols = [c for c in result.columns if c.endswith(("_haplotype", "_ns_changes"))]
        fixed_cols = {"haplotype_id", "n_samples", "combination_ids"} | set(hap_cols)
        per_pos_cols = [
            c for c in result.columns
            if c not in fixed_cols and not c.lstrip("-").isdigit()
        ]

        # ── Haplotype summary ──────────────────────────────────────────────
        st.subheader("Haplotype summary")
        summary_cols = ["haplotype_id", "n_samples", "combination_ids"] + hap_cols + per_pos_cols
        st.dataframe(
            result[[c for c in summary_cols if c in result.columns]],
            width="stretch", hide_index=True,
        )

        # ── Per-sample haplotypes ──────────────────────────────────────────
        _fixed_raw = {"combination_id", "n_samples", "sample_ids"}
        _hap_raw   = {c for c in raw.columns if c.endswith(("_haplotype", "_ns_changes"))}
        # Exclude raw allele-matrix position columns (pure numeric strings like "139150");
        # keep only the named per-range columns added by compute_haplotypes.
        per_pos_raw = [
            c for c in raw.columns
            if c not in _fixed_raw and c not in _hap_raw and not c.lstrip("-").isdigit()
        ]

        if per_pos_raw:
            per_sample = (
                raw[["combination_id", "sample_ids"] + per_pos_raw]
                .explode("sample_ids")
                .rename(columns={"sample_ids": "sample_id"})
                .reset_index(drop=True)
            )
            st.subheader("Per-sample haplotypes")
            st.dataframe(per_sample, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Debug
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Debug"):
    if genotypes_loaded:
        allele_matrix_debug = build_allele_matrix(
            regions,
            excluded_positions=excluded_positions,
            het_mode=HET_MODE,
        )
        if "sample_to_id" in st.session_state:
            dm = allele_matrix_debug.copy()
            dm.insert(0, "combination_id", dm.index.map(st.session_state["sample_to_id"]))
        else:
            dm = allele_matrix_debug
        st.write("**Allele matrix (genotypes):**")
        st.dataframe(dm, width="stretch")

    if "haplotype_raw" in st.session_state:
        st.write("**Haplotype combinations (raw):**")
        st.dataframe(st.session_state["haplotype_raw"], width="stretch", hide_index=True)

    st.write("**Parsed loci:**")
    st.dataframe(loci_df, width="stretch", hide_index=True)
    st.write("**Resolved genomic intervals:**")
    for sid, info in resolved.items():
        st.text(f"{sid} ({info['coord_type']}): {info['intervals']}")
