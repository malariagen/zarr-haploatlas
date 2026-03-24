import time
import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, resolve_loci,
    query_locus_metadata, load_genotypes,
)

st.set_page_config(layout="wide", page_title="Pf Haplotype Explorer")

# ── Load static data ──────────────────────────────────────────────────────────
chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── User input ────────────────────────────────────────────────────────────────
RAW_USER_INPUT = st.text_input(
    "Enter genomic loci",
    value="PF3D7_0709000[72-76,326] Pf3D7_04_v3[104205,139150-139156]",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Multiple loci separated by spaces."
    ),
)

# Clear stale session state on input change
if st.session_state.get("last_input") != RAW_USER_INPUT:
    for k in [k for k in st.session_state if k.startswith((
        "genotypes_", "_region_xr_", "haplotype_df",
    ))]:
        del st.session_state[k]
    st.session_state["last_input"] = RAW_USER_INPUT

loci_df  = parse_loci_from_input(RAW_USER_INPUT)
resolved = resolve_loci(loci_df, reference_files["exon_gff"])

if loci_df.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Unified variant metadata table
# ══════════════════════════════════════════════════════════════════════════════
st.header("Variant Sites")

# Variants in Pf can have long REF alleles (due to nearby indels), so a
# variant starting upstream of the queried interval may still affect it.
# Pad the query window and post-filter by actual REF footprint overlap.
UPSTREAM_PAD = 100   # bp

all_meta = {}   # source_id → filtered region_meta
meta_rows = []

for source_id, locus_info in resolved.items():
    intervals = locus_info["intervals"]
    if not intervals:
        st.warning(f"No genomic intervals resolved for `{source_id}`")
        continue

    # Pad intervals upstream for the zarr query
    padded_intervals = tuple(
        (chrom, max(1, start - UPSTREAM_PAD), end)
        for chrom, start, end in intervals
    )

    # NOT cached — re-runs each time, freshly sets _region_xr_ in session state
    region_meta = query_locus_metadata(
        variant_data, chunk_index_df,
        source_id, padded_intervals,
    )
    if region_meta is None:
        st.warning(f"No variants found for `{source_id}`")
        continue

    # Post-filter: keep only variants whose REF footprint overlaps at least
    # one original (unpadded) interval.
    # A variant at POS with REF length L covers [POS, POS+L-1].
    positions = region_meta["positions"]
    alleles   = region_meta["alleles"]
    keep_mask = np.zeros(len(positions), dtype=bool)

    for vi in range(len(positions)):
        vpos    = int(positions[vi])
        ref_len = len(str(alleles[vi][0]))
        v_end   = vpos + ref_len - 1

        for chrom, iv_start, iv_end in intervals:
            if vpos <= iv_end and v_end >= iv_start:
                keep_mask[vi] = True
                break

    if not keep_mask.any():
        st.warning(f"No variants found for `{source_id}`")
        continue

    keep_idx = np.where(keep_mask)[0]

    # Slice metadata arrays
    filtered_meta = {
        "positions":   positions[keep_idx],
        "alleles":     alleles[keep_idx],
        "sample_ids":  region_meta["sample_ids"],
        "is_snp":      region_meta["is_snp"][keep_idx],
        "filter_pass": region_meta["filter_pass"][keep_idx],
        "CDS":         region_meta["CDS"][keep_idx],
        "numalt":      region_meta["numalt"][keep_idx],
        "n_variants":  int(keep_idx.shape[0]),
        "n_samples":   region_meta["n_samples"],
    }

    # Slice the xarray in session state so load_genotypes stays in sync
    xr_key = f"_region_xr_{source_id}"
    st.session_state[xr_key] = st.session_state[xr_key].isel(variants=keep_idx)

    all_meta[source_id] = filtered_meta

    # Build display rows
    coord_type  = locus_info["coord_type"]
    locus_chrom = intervals[0][0] if intervals else ""

    for vi in range(filtered_meta["n_variants"]):
        ref_allele = str(filtered_meta["alleles"][vi][0])
        alts = [str(a) for a in filtered_meta["alleles"][vi][1:] if str(a).strip()]
        meta_rows.append({
            "chrom":             locus_chrom,
            "position":          filtered_meta["positions"][vi],
            "source":            source_id,
            "needs_translation": coord_type == "aa",
            "ref":               ref_allele,
            "ref_len":           len(ref_allele),
            "alt":               ", ".join(alts),
            "is_snp":            filtered_meta["is_snp"][vi],
            "filter_pass":       filtered_meta["filter_pass"][vi],
            "CDS":               filtered_meta["CDS"][vi],
            "numalt":            filtered_meta["numalt"][vi],
        })

if not all_meta:
    st.warning("No variants found for any queried locus.")
    st.stop()

meta_df = pd.DataFrame(meta_rows)
st.dataframe(meta_df, use_container_width=True, hide_index=True)

total_variants = sum(m["n_variants"] for m in all_meta.values())
n_samples = next(iter(all_meta.values()))["n_samples"]
st.caption(
    f"{total_variants} variant sites across {len(all_meta)} loci · "
    f"{n_samples:,} samples · "
    f"Queried with {UPSTREAM_PAD}bp upstream padding to capture "
    f"variants with long REF alleles spanning into the region."
)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load genotypes (all loci at once)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Genotypes")

geno_key = "genotypes_all"
st.session_state.setdefault(geno_key, None)

if st.session_state[geno_key] is None:
    st.info(f"Click below to load genotypes for {n_samples:,} samples across all loci.")
    if st.button("Load genotypes", type="primary"):
        t0 = time.time()
        with st.spinner(f"Fetching genotypes for {n_samples:,} samples…"):
            loaded = {}
            for source_id in all_meta:
                loaded[source_id] = load_genotypes(source_id)
            st.session_state[geno_key] = loaded
        st.toast(f"✅ Loaded in {time.time() - t0:.1f}s")
        st.rerun()

if st.session_state[geno_key] is not None:
    all_genotypes = st.session_state[geno_key]
    sample_ids = next(iter(all_meta.values()))["sample_ids"]

    # ── Per-sample unified view ───────────────────────────────────────────
    st.subheader("Per-Sample View")
    selected = st.selectbox(
        "Select sample:",
        options=range(n_samples),
        format_func=lambda i: f"{i+1}/{n_samples}: {sample_ids[i]}",
    )

    rows = []
    for source_id, meta in all_meta.items():
        gt = all_genotypes[source_id][:, selected, :]   # (n_variants, 2)
        alleles = meta["alleles"]
        positions = meta["positions"]
        locus_info = resolved[source_id]
        locus_chrom = locus_info["intervals"][0][0] if locus_info["intervals"] else ""

        for vi in range(meta["n_variants"]):
            g = gt[vi]
            ref = str(alleles[vi][0])
            if g[0] < 0 or g[1] < 0:
                a1 = a2 = "?"
            else:
                a1 = str(alleles[vi][g[0]])
                a2 = str(alleles[vi][g[1]])
            rows.append({
                "chrom":    locus_chrom,
                "position": positions[vi],
                "source":   source_id,
                "ref":      ref,
                "allele_1": a1,
                "allele_2": a2,
                "is_het":   a1 != a2 and a1 != "?",
            })

    gt_df = pd.DataFrame(rows)
    st.dataframe(gt_df, use_container_width=True, hide_index=True)

    # ── Simple haplotype counts (cross-locus) ─────────────────────────────
    st.divider()
    st.subheader("Haplotype Counts")

    if st.button("Compute haplotype counts"):
        t0 = time.time()
        with st.spinner("Computing…"):
            hap_parts = []
            for source_id, meta in all_meta.items():
                gt = all_genotypes[source_id]
                alleles = meta["alleles"]

                for vi in range(meta["n_variants"]):
                    col_vals = []
                    for si in range(n_samples):
                        g = gt[vi, si, :]
                        if g[0] < 0 or g[1] < 0:
                            col_vals.append("?")
                        else:
                            a0 = str(alleles[vi][g[0]])
                            a1 = str(alleles[vi][g[1]])
                            if a0 != a1:
                                col_vals.append(f"{a0}*")
                            else:
                                col_vals.append(a0)
                    hap_parts.append(col_vals)

            hap_strings = []
            for si in range(n_samples):
                parts = [hap_parts[vi][si] for vi in range(len(hap_parts))]
                hap_strings.append("-".join(parts))

            counts = pd.Series(hap_strings).value_counts().reset_index()
            counts.columns = ["haplotype", "count"]
            counts["frequency"] = counts["count"] / n_samples

        st.session_state["haplotype_df"] = counts
        st.toast(f"✅ Computed in {time.time() - t0:.2f}s")
        st.rerun()

    if "haplotype_df" in st.session_state and st.session_state["haplotype_df"] is not None:
        hap_df = st.session_state["haplotype_df"]
        st.write(f"**{len(hap_df)}** unique haplotypes")
        st.dataframe(
            hap_df.head(20).style.format({"frequency": "{:.2%}"}),
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
    st.write("**Chunk index:**")
    st.dataframe(chunk_index_df)