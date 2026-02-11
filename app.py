import malariagen_data
import streamlit as st
import pandas as pd
import xarray as xr

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

def query_region(chunk_df, chrom, start_pos, end_pos, variant_data):
    """Load only the chunks needed for a region and filter to exact positions"""
    matching_chunks = find_chunks_for_region(chunk_df, chrom, start_pos, end_pos)
    
    if len(matching_chunks) == 0:
        return None
    
    # Get unique chunk IDs
    chunk_ids = matching_chunks['chunk'].unique()
    
    # Load and combine chunks
    chunk_datasets = [load_chunk(chunk_df, cid, variant_data) for cid in chunk_ids]
    
    if len(chunk_datasets) == 1:
        region_data = chunk_datasets[0]
    else:
        region_data = xr.concat(chunk_datasets, dim='variants')
    
    # Filter to exact chromosome and position range
    mask = (
        (region_data["variant_chrom"].values == chrom) &
        (region_data["variant_position"].values >= start_pos) &
        (region_data["variant_position"].values <= end_pos)
    )
    
    return region_data.isel(variants=mask)

# Load data
variant_data = load_full_variant_data()
chunk_df = zarr_chunk_metadata()

# User input
user_input = st.text_input("Insert region:", value="Pf3D7_07_v3:401,000-401,100")
chrom, start_pos, end_pos = extract_locus(user_input)

# Find matching chunks
matching_chunks = find_chunks_for_region(chunk_df, chrom, start_pos, end_pos)

# Query region
region = query_region(chunk_df, chrom, start_pos, end_pos, variant_data)

# Display results
st.header("Results")

if region is not None:
    n_variants = region.dims['variants']
    st.write(f"Found **{n_variants}** variants in `{chrom}:{start_pos:,}-{end_pos:,}`")
    
    # Show variant summary
    if n_variants > 0:
        positions = region["variant_position"].values
        st.write(f"Position range: {positions.min():,} - {positions.max():,}")
        
        # Show as dataframe
        variant_df = pd.DataFrame({
            'position': region["variant_position"].values,
            'is_snp': region["variant_is_snp"].values,
            'filter_pass': region["variant_filter_pass"].values,
            'CDS': region["variant_CDS"].values,
            'numalt': region["variant_numalt"].values,
        })
        st.dataframe(variant_df)
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
    st.divider()