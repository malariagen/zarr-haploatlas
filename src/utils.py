import streamlit as st
import numpy as np
import pandas as pd
import malariagen_data
import re

@st.cache_data
def load_reference_files():
    def parse_attributes(attr_string):
        attrs = {}
        for item in attr_string.split(';'):
            if '=' in item:
                key, value = item.split('=', 1)
                attrs[key] = value
        return attrs
    
    gff = pd.read_csv(
        "assets/PlasmoDB-54_Pfalciparum3D7.gff",
        sep="\t", comment="#", header=None,
        names=["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    )

    attr_df = gff["attributes"].apply(parse_attributes).apply(pd.Series)
    gff = (
        pd.concat([gff, attr_df], axis=1)
        .drop(columns = ["source", "attributes", "Note", "score", "protein_source_id"])
        .sort_values("gene_id")
    )

    return {
        "exon_gff": gff[gff["type"] == "exon"].reset_index(drop = True)
    }

@st.cache_data(show_spinner="Building chunk index from zarr...", persist="disk")
def build_chunk_index() -> pd.DataFrame:
    pf8 = malariagen_data.Pf8()
    variant_data = pf8.variant_calls()

    pos_var = variant_data["variant_position"]
    chrom_var = variant_data["variant_chrom"]
    chunk_sizes = pos_var.chunks[0]

    chunk_index = []
    offset = 0
    n_chunks = len(chunk_sizes)

    progress = st.progress(0, text="Processing chunks...")

    for i, chunk_size in enumerate(chunk_sizes):
        chunk_start = offset
        chunk_end = offset + chunk_size

        chunk_pos = pos_var.isel(variants=slice(chunk_start, chunk_end)).values
        chunk_chrom = chrom_var.isel(variants=slice(chunk_start, chunk_end)).values

        for chrom in np.unique(chunk_chrom):
            chrom_mask = chunk_chrom == chrom
            chrom_positions = chunk_pos[chrom_mask]
            chrom_indices = np.where(chrom_mask)[0]

            chunk_index.append({
                "chunk":       i,
                "chunk_start": chunk_start + chrom_indices[0],
                "chunk_end":   chunk_start + chrom_indices[-1] + 1,
                "chrom":       chrom,
                "min_pos":     int(chrom_positions.min()),
                "max_pos":     int(chrom_positions.max()),
            })

        offset += chunk_size
        progress.progress((i + 1) / n_chunks, text=f"Chunk {i + 1} / {n_chunks}")

    progress.empty()
    return pd.DataFrame(chunk_index)

def parse_loci_from_input(user_input: str) -> pd.DataFrame:
    rows = []

    for part in user_input.split():
        if "[" not in part or "]" not in part:
            continue

        identifier, pos_part = part.split("[", 1)
        pos_part = pos_part.rstrip("]")

        is_gene = bool(re.match(r"^PF3D7_\d{7}$", identifier, re.IGNORECASE))
        coord_type = "aa" if is_gene else "nt"

        for token in pos_part.split(","):
            token = token.strip()
            if "-" in token:
                start, end = token.split("-", 1)
                rows.append((identifier, int(start), int(end), coord_type))
            else:
                pos = int(token)
                rows.append((identifier, pos, pos, coord_type))

    return pd.DataFrame(rows, columns=["chrom", "start", "end", "coord_type"])