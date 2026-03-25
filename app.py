import time
import pandas as pd
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, resolve_loci,
    query_locus_metadata, filter_region_by_intervals,
    build_variant_rows, load_genotypes, build_allele_matrix,
)
from src.haplotypes import deduplicate_allele_matrix, compute_haplotypes

st.set_page_config(layout="wide", page_title="Pf Haplotype Explorer")

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
    st.session_state["last_input"] = RAW_USER_INPUT

loci_df  = parse_loci_from_input(RAW_USER_INPUT)
resolved = resolve_loci(loci_df, reference_files["cds_gff"])

if loci_df.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Variant metadata table
# ══════════════════════════════════════════════════════════════════════════════

# Variants in Pf can have long REF alleles (due to nearby indels), so a
# variant starting upstream of the queried interval may still affect it.
# Pad the query window and post-filter by actual REF footprint overlap.
UPSTREAM_PAD = 100  # bp

# regions: source_id → {"meta": dict, "ds": xr.Dataset, "genotypes": ndarray|None}
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
        regions[source_id] = {"meta": meta, "ds": ds, "genotypes": None}

    st.session_state["regions"] = regions

regions = st.session_state["regions"]

if not regions:
    st.warning("No variants found for any queried locus.")
    st.stop()

st.header("Variant Sites")
meta_rows = []
for source_id, region in regions.items():
    meta_rows.extend(build_variant_rows(source_id, region["meta"], resolved[source_id]))

meta_df     = pd.DataFrame(meta_rows)
n_samples   = next(iter(regions.values()))["meta"]["n_samples"]
total_vars  = sum(r["meta"]["n_variants"] for r in regions.values())

st.dataframe(meta_df, use_container_width=True, hide_index=True)
st.caption(
    f"{total_vars} variant sites across {len(regions)} loci · "
    f"{n_samples:,} samples · "
    f"Queried with {UPSTREAM_PAD}bp upstream padding to capture "
    f"variants with long REF alleles spanning into the region."
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load genotypes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Genotypes")

genotypes_loaded = all(r["genotypes"] is not None for r in regions.values())

if not genotypes_loaded:
    st.info(f"Click below to load genotypes for {n_samples:,} samples across all loci.")
    if st.button("Load genotypes", type="primary"):
        t0 = time.time()
        with st.spinner(f"Fetching genotypes for {n_samples:,} samples…"):
            for region in regions.values():
                region["genotypes"] = load_genotypes(region["ds"])
        st.toast(f"✅ Loaded in {time.time() - t0:.1f}s")
else:
    allele_matrix = build_allele_matrix(regions)
    st.dataframe(allele_matrix, use_container_width=True)

    # ── Haplotypes ────────────────────────────────────────────────────────
    st.divider()
    st.header("Haplotypes")

    if st.button("Compute haplotypes", type="primary"):
        t0 = time.time()
        with st.spinner("Computing haplotypes…"):
            deduped = deduplicate_allele_matrix(allele_matrix)
            haplotype_result = compute_haplotypes(
                deduped, regions, resolved, loci_df, reference_files["cds_gff"]
            )
            st.session_state["haplotype_result"] = haplotype_result
        st.toast(f"✅ Computed in {time.time() - t0:.2f}s")

    if "haplotype_result" in st.session_state:
        result  = st.session_state["haplotype_result"]
        hap_cols = [c for c in result.columns
                    if c.endswith(("_haplotype", "_ns_changes"))]
        st.dataframe(
            result[["n_samples"] + hap_cols].sort_values("n_samples", ascending=False),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Debug
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("Inner workings"):
    st.write("**Parsed loci:**")
    st.dataframe(loci_df, hide_index=True)
    st.write("**Resolved genomic intervals:**")
    for sid, info in resolved.items():
        st.text(f"{sid} ({info['coord_type']}): {info['intervals']}")
