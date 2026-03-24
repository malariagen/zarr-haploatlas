import time
import pandas as pd
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, resolve_loci,
    query_locus_metadata, load_genotypes,
    get_sample_genotypes, get_haplotype_counts,
)

st.set_page_config(layout="wide")

# ── Load static data ──────────────────────────────────────────────────────────
chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── User input ────────────────────────────────────────────────────────────────
RAW_USER_INPUT = st.text_input(
    "Enter a genomic region",
    value="PF3D7_0709000[72-76,326] Pf3D7_04_v3[104205,139150-139156]",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Multiple loci separated by spaces."
    ),
)

loci_df  = parse_loci_from_input(RAW_USER_INPUT)
resolved = resolve_loci(loci_df, reference_files["exon_gff"])

# ── Results ───────────────────────────────────────────────────────────────────
st.header("Results")

for source_id, locus_info in resolved.items():
    intervals  = locus_info["intervals"]
    coord_type = locus_info["coord_type"]
    label      = "amino acid" if coord_type == "aa" else "nucleotide"

    with st.container(border=True):
        st.subheader(f"`{source_id}`  —  {label}")

        if not intervals:
            st.warning(f"No genomic intervals resolved for `{source_id}`")
            continue

        region_meta = query_locus_metadata(
            variant_data, chunk_index_df,
            source_id, tuple(intervals),
        )

        if region_meta is None:
            st.warning(f"No variants found for `{source_id}`")
            continue

        n_variants = region_meta["n_variants"]
        n_samples  = region_meta["n_samples"]
        st.write(f"Found **{n_variants}** variants across **{n_samples:,}** samples")

        if n_variants == 0:
            continue

        # Variant table
        st.dataframe(
            pd.DataFrame({
                "position":    region_meta["positions"],
                "is_snp":      region_meta["is_snp"],
                "filter_pass": region_meta["filter_pass"],
                "CDS":         region_meta["CDS"],
                "numalt":      region_meta["numalt"],
            }),
            use_container_width=True, hide_index=True,
        )

        st.divider()

        # ── Genotype loading ──────────────────────────────────────────────────
        geno_key    = f"genotypes_{source_id}"
        haplo_key   = f"haplotype_df_{source_id}"
        sample_key  = f"sample_idx_{source_id}"

        st.session_state.setdefault(geno_key,   None)
        st.session_state.setdefault(haplo_key,  None)
        st.session_state.setdefault(sample_key, 0)

        if st.session_state[geno_key] is None:
            st.info(f"Genotypes not loaded. Click below to fetch data for {n_samples:,} samples.")
            if st.button("Load genotypes", key=f"load_geno_{source_id}", type="primary"):
                t0 = time.time()
                with st.spinner(f"Fetching genotypes for {n_samples:,} samples..."):
                    st.session_state[geno_key] = load_genotypes(source_id)
                st.toast(f"✅ Loaded in {time.time() - t0:.1f}s")
                st.rerun()

        if st.session_state[geno_key] is not None:
            genotypes  = st.session_state[geno_key]
            sample_ids = region_meta["sample_ids"]

            selected = st.selectbox(
                "Select sample:",
                options=range(n_samples),
                format_func=lambda i: f"{i + 1}/{n_samples}: {sample_ids[i]}",
                index=st.session_state[sample_key],
                key=f"sample_select_{source_id}",
            )
            st.session_state[sample_key] = selected

            t0    = time.time()
            gt_df = get_sample_genotypes(region_meta, genotypes, selected)
            st.dataframe(gt_df, use_container_width=True, hide_index=True)
            st.caption(f"⏱️ {time.time() - t0:.3f}s")

            # ── Haplotype distribution ────────────────────────────────────────
            st.divider()
            st.subheader("Haplotype Distribution")

            if st.button("Compute haplotype counts", key=f"haplo_btn_{source_id}"):
                t0 = time.time()
                with st.spinner("Computing haplotype frequencies..."):
                    st.session_state[haplo_key] = get_haplotype_counts(region_meta, genotypes)
                st.toast(f"✅ Computed in {time.time() - t0:.2f}s")
                st.rerun()

            if st.session_state[haplo_key] is not None:
                haplotype_df = st.session_state[haplo_key]
                st.write(f"**{len(haplotype_df)}** unique haplotypes from **{n_samples:,}** samples")
                st.dataframe(
                    haplotype_df.head(10).style.format({"frequency": "{:.2%}"}),
                    use_container_width=True, hide_index=True,
                )