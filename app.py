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

def load_chunk(chunk_df, chunk_id, variant_data):
    """Load a single chunk as an xarray Dataset"""
    row = chunk_df[chunk_df['chunk'] == chunk_id].iloc[0]
    return variant_data.isel(variants=slice(int(row['slice_start']), int(row['slice_end'])))

def query_region_lazy(chunk_df, chrom, start_pos, end_pos, variant_data):
    """Load region metadata only (fast) - no genotypes"""
    matching_chunks = find_chunks_for_region(chunk_df, chrom, start_pos, end_pos)
    
    if len(matching_chunks) == 0:
        return None
    
    chunk_ids = matching_chunks['chunk'].unique()
    chunk_datasets = [load_chunk(chunk_df, cid, variant_data) for cid in chunk_ids]
    
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
    
    # Preload only lightweight metadata
    result.attrs['_positions'] = result["variant_position"].values
    alleles = result["variant_allele"].values
    if alleles.dtype.kind == 'S':
        alleles = alleles.astype('U1')
    result.attrs['_alleles'] = alleles
    result.attrs['_sample_ids'] = result["sample_id"].values
    result.attrs['_genotypes_loaded'] = False
    
    return result

def load_genotypes(region):
    """Load genotypes into region (slow network call)"""
    region.attrs['_genotypes'] = region["call_genotype"].values.astype(np.int8)
    region.attrs['_genotypes_loaded'] = True
    return region

def get_sample_genotypes(region, sample_idx):
    """Get genotypes for a single sample - uses cached data"""
    positions = region.attrs['_positions']
    alleles = region.attrs['_alleles']
    genotypes = region.attrs['_genotypes'][:, sample_idx, :]
    
    n_variants = len(positions)
    gt1 = genotypes[:, 0]
    gt2 = genotypes[:, 1]
    
    row_idx = np.arange(n_variants)
    valid1 = gt1 >= 0
    valid2 = gt2 >= 0
    
    allele_1 = np.full(n_variants, '.', dtype='U1')
    allele_2 = np.full(n_variants, '.', dtype='U1')
    
    allele_1[valid1] = alleles[row_idx[valid1], gt1[valid1]]
    allele_2[valid2] = alleles[row_idx[valid2], gt2[valid2]]
    
    genotype = f"{gt1[0]}/{gt2[0]}"
    
    return pd.DataFrame({
        'position': positions,
        'genotype': genotype,
        'allele_1': allele_1,
        'allele_2': allele_2,
    })

def get_all_sample_alleles(region):
    """Return allele calls as two separate DataFrames - optimized"""
    positions = region.attrs['_positions']
    alleles = region.attrs['_alleles']
    sample_ids = region.attrs['_sample_ids']
    genotypes = region.attrs['_genotypes']
    
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
    
    df1 = pd.DataFrame(allele_1.astype('U1'), index=positions, columns=sample_ids)
    df2 = pd.DataFrame(allele_2.astype('U1'), index=positions, columns=sample_ids)
    
    df1.index.name = 'position'
    df2.index.name = 'position'
    
    return df1, df2

# Load data
variant_data = load_full_variant_data()
chunk_df = zarr_chunk_metadata()

# User input
user_input = st.text_input("Insert region:", value="Pf3D7_07_v3:401,000-401,100")
chrom, start_pos, end_pos = extract_locus(user_input)

# Query region metadata (fast)
matching_chunks = find_chunks_for_region(chunk_df, chrom, start_pos, end_pos)
region = query_region_lazy(chunk_df, chrom, start_pos, end_pos, variant_data)

# Display results
st.header("Results")

if region is not None:
    n_variants = region.dims['variants']
    n_samples = region.dims['samples']
    st.write(f"Found **{n_variants}** variants in `{chrom}:{start_pos:,}-{end_pos:,}`")
    
    if n_variants > 0:
        # Variant info table
        variant_df = pd.DataFrame({
            'position': region.attrs['_positions'],
            'is_snp': region["variant_is_snp"].values,
            'filter_pass': region["variant_filter_pass"].values,
            'CDS': region["variant_CDS"].values,
            'numalt': region["variant_numalt"].values,
        })
        st.dataframe(variant_df, width="stretch", hide_index=True)
        
        st.divider()
        st.subheader("Sample Genotypes")
        
        # Load genotypes button
        if 'region_with_gt' not in st.session_state:
            st.session_state.region_with_gt = None
        
        if st.session_state.region_with_gt is None:
            st.info(f"Genotypes not loaded. Click below to fetch data for {n_samples:,} samples.")
            if st.button("Load genotypes", type="primary"):
                start = time.time()
                
                with st.spinner(f"Fetching genotypes for {n_samples:,} samples... (this may take a few minutes)"):
                    region = load_genotypes(region)
                    st.session_state.region_with_gt = region
                
                st.toast(f"Loaded in {time.time() - start:.1f}s")
                st.rerun()
        else:
            region = st.session_state.region_with_gt
            
            # Sample selector
            sample_ids = region.attrs['_sample_ids']
            
            if 'sample_idx' not in st.session_state:
                st.session_state.sample_idx = 0
            
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
            gt_df = get_sample_genotypes(region, st.session_state.sample_idx)
            st.dataframe(gt_df, width="stretch", hide_index=True)
            st.caption(f"⏱️ Single sample: {time.time() - start:.3f}s")
            
            # Load all samples button
            st.divider()
            st.subheader("All Samples")
            
            if st.button("Show all samples"):
                start = time.time()
                allele_1_df, allele_2_df = get_all_sample_alleles(region)
                
                st.caption(f"⏱️ All samples: {time.time() - start:.3f}s")
                st.write(f"Shape: {allele_1_df.shape[0]} positions × {allele_1_df.shape[1]} samples")

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