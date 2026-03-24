import malariagen_data
import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import time

@st.cache_data
def load_full_variant_data():
    release_data = malariagen_data.Pf8()
    variant_data = release_data.variant_calls()
    return variant_data

@st.cache_data
def zarr_chunk_metadata():
    return pd.read_csv("zarr-chunk-meta.tsv", sep="\t")

def extract_locus(user_input: str):
    chrom, pos = user_input.split(":")
    start, end = pos.split("-")
    return chrom, int(start.replace(",", "")), int(end.replace(",", ""))

def find_chunks_for_region(chunk_df, chrom, start_pos, end_pos):
    """Return chunk entries that overlap the given region"""
    return chunk_df[
        (chunk_df['chrom'] == chrom) &
        (chunk_df['max_pos'] >= start_pos) &
        (chunk_df['min_pos'] <= end_pos)
    ]

@st.cache_data
def query_region_metadata(_variant_data, _chunk_df, chrom, start_pos, end_pos):
    """Load region metadata only (cached)"""
    matching_chunks = find_chunks_for_region(_chunk_df, chrom, start_pos, end_pos)
    
    if len(matching_chunks) == 0:
        return None, matching_chunks
    
    chunk_ids = matching_chunks['chunk'].unique()
    
    chunk_datasets = []
    for cid in chunk_ids:
        row = _chunk_df[_chunk_df['chunk'] == cid].iloc[0]
        chunk_datasets.append(_variant_data.isel(variants=slice(int(row['slice_start']), int(row['slice_end']))))
    
    if len(chunk_datasets) == 1:
        region_data = chunk_datasets[0]
    else:
        region_data = xr.concat(chunk_datasets, dim='variants')
    
    mask = (
        (region_data["variant_chrom"].values == chrom) &
        (region_data["variant_position"].values >= start_pos) &
        (region_data["variant_position"].values <= end_pos)
    )
    
    result = region_data.isel(variants=mask)
    
    # Return serializable metadata dict
    positions = result["variant_position"].values
    alleles = result["variant_allele"].values
    if alleles.dtype.kind == 'S':
        alleles = alleles.astype('U1')
    
    region_meta = {
        'positions': positions,
        'alleles': alleles,
        'sample_ids': result["sample_id"].values,
        'is_snp': result["variant_is_snp"].values,
        'filter_pass': result["variant_filter_pass"].values,
        'CDS': result["variant_CDS"].values,
        'numalt': result["variant_numalt"].values,
        'n_variants': result.dims['variants'],
        'n_samples': result.dims['samples'],
    }
    
    # Store region reference in session state for genotype loading
    st.session_state._region_xr = result
    
    return region_meta, matching_chunks

def load_genotypes():
    """Load genotypes from cached xarray region"""
    region = st.session_state._region_xr
    return region["call_genotype"].values.astype(np.int8)

def get_sample_genotypes(region_meta, genotypes, sample_idx):
    """Get genotypes for a single sample"""
    positions = region_meta['positions']
    alleles = region_meta['alleles']
    gt = genotypes[:, sample_idx, :]
    
    n_variants = len(positions)
    gt1 = gt[:, 0]
    gt2 = gt[:, 1]
    
    row_idx = np.arange(n_variants)
    valid1 = gt1 >= 0
    valid2 = gt2 >= 0
    
    allele_1 = np.full(n_variants, '.', dtype='U1')
    allele_2 = np.full(n_variants, '.', dtype='U1')
    
    allele_1[valid1] = alleles[row_idx[valid1], gt1[valid1]]
    allele_2[valid2] = alleles[row_idx[valid2], gt2[valid2]]
    
    return pd.DataFrame({
        'position': positions,
        'allele_1': allele_1,
        'allele_2': allele_2,
    })

def get_haplotype_counts(region_meta, genotypes):
    """Get haplotype distribution across all samples"""
    positions = region_meta['positions']
    alleles = region_meta['alleles']
    
    n_variants, n_samples, _ = genotypes.shape
    
    gt1 = genotypes[:, :, 0]
    gt2 = genotypes[:, :, 1]
    
    row_idx = np.arange(n_variants)[:, np.newaxis]
    
    alleles_bytes = alleles.astype('S1') if alleles.dtype.kind == 'U' else alleles
    
    gt1_clipped = np.clip(gt1, 0, alleles.shape[1] - 1)
    gt2_clipped = np.clip(gt2, 0, alleles.shape[1] - 1)
    
    allele_1 = alleles_bytes[row_idx, gt1_clipped]
    allele_2 = alleles_bytes[row_idx, gt2_clipped]
    
    allele_1 = np.where(gt1 >= 0, allele_1, b'.')
    allele_2 = np.where(gt2 >= 0, allele_2, b'.')
    
    # Build haplotype strings: transpose to (n_samples, n_variants)
    allele_1_T = allele_1.T.astype('U1')
    allele_2_T = allele_2.T.astype('U1')
    
    haplotype_1 = ['-'.join(row) for row in allele_1_T]
    haplotype_2 = ['-'.join(row) for row in allele_2_T]
    
    all_haplotypes = haplotype_1 + haplotype_2
    
    counts = pd.Series(all_haplotypes).value_counts().reset_index()
    counts.columns = ['haplotype', 'count']
    counts['count'] = counts['count'] // 2
    counts['frequency'] = counts['count'] / len(all_haplotypes) * 2
    
    return counts

# Load data
variant_data = load_full_variant_data()
chunk_df = zarr_chunk_metadata()

# User input
user_input = st.text_input("Insert region:", value="Pf3D7_07_v3:401,000-401,100")
chrom, start_pos, end_pos = extract_locus(user_input)

# Query region metadata (cached)
region_meta, matching_chunks = query_region_metadata(variant_data, chunk_df, chrom, start_pos, end_pos)

# Display results
st.header("Results")

if region_meta is not None:
    n_variants = region_meta['n_variants']
    n_samples = region_meta['n_samples']
    st.write(f"Found **{n_variants}** variants in `{chrom}:{start_pos:,}-{end_pos:,}`")
    
    if n_variants > 0:
        # Variant info table
        variant_df = pd.DataFrame({
            'position': region_meta['positions'],
            'is_snp': region_meta['is_snp'],
            'filter_pass': region_meta['filter_pass'],
            'CDS': region_meta['CDS'],
            'numalt': region_meta['numalt'],
        })
        st.dataframe(variant_df, width="stretch", hide_index=True)
        
        st.divider()
        st.subheader("Sample Genotypes")
        
        # Initialize session state
        if 'genotypes' not in st.session_state:
            st.session_state.genotypes = None
        if 'sample_idx' not in st.session_state:
            st.session_state.sample_idx = 0
        if 'haplotype_df' not in st.session_state:
            st.session_state.haplotype_df = None
        
        genotypes_loaded = st.session_state.genotypes is not None
        
        if not genotypes_loaded:
            st.info(f"Genotypes not loaded. Click below to fetch data for {n_samples:,} samples.")
            if st.button("Load genotypes", type="primary"):
                start = time.time()
                with st.spinner(f"Fetching genotypes for {n_samples:,} samples..."):
                    st.session_state.genotypes = load_genotypes()
                    genotypes_loaded = True
                st.toast(f"✅ Loaded in {time.time() - start:.1f}s")
        
        if genotypes_loaded:
            genotypes = st.session_state.genotypes
            sample_ids = region_meta['sample_ids']
            
            selected_sample = st.selectbox(
                "Select sample:",
                options=range(n_samples),
                format_func=lambda i: f"{i+1}/{n_samples}: {sample_ids[i]}",
                index=st.session_state.sample_idx,
                key="sample_select"
            )
            st.session_state.sample_idx = selected_sample
            
            # Display genotypes for selected sample
            start = time.time()
            gt_df = get_sample_genotypes(region_meta, genotypes, st.session_state.sample_idx)
            st.dataframe(gt_df, width="stretch", hide_index=True)
            st.caption(f"⏱️ Time taken to change sample: {time.time() - start:.3f}s")
            
            # Haplotype distribution
            st.divider()
            st.subheader("Haplotype Distribution")
            
            if st.button("Show haplotype counts"):
                start = time.time()
                with st.spinner("Computing haplotype frequencies..."):
                    st.session_state.haplotype_df = get_haplotype_counts(region_meta, genotypes)
                st.toast(f"✅ Computed in {time.time() - start:.2f}s")
            
            if st.session_state.haplotype_df is not None:
                haplotype_df = st.session_state.haplotype_df
                
                st.write(f"**{len(haplotype_df)}** unique haplotypes from **{n_samples:,}** samples")
                
                st.dataframe(
                    haplotype_df.head(10).style.format({'frequency': '{:.2%}'}),
                    width="stretch",
                    hide_index=True
                )

else:
    st.warning("No variants found in this region")

# Debug info
with st.expander("See inner workings"):
    st.write("Parsed locus:")
    st.text(f"Chromosome: {chrom}")
    st.text(f"Start: {start_pos:,}")
    st.text(f"End: {end_pos:,}")
    st.divider()
    
    st.write("Matching chunks:")
    st.dataframe(matching_chunks)
    st.divider()
    
    st.write("Full chunk metadata:")
    st.dataframe(chunk_df)