import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import malariagen_data
import re


# ── Reference data ────────────────────────────────────────────────────────────

@st.cache_data
def load_reference_files() -> dict:
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
        .drop(columns=["source", "attributes", "Note", "score", "protein_source_id"])
        .sort_values("gene_id")
    )
    return {
        "exon_gff": gff[gff["type"] == "exon"].reset_index(drop=True)
    }


@st.cache_data
def load_variant_data():
    return malariagen_data.Pf8().variant_calls()


# ── Chunk index ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Building chunk index from zarr...", persist="disk")
def build_chunk_index() -> pd.DataFrame:
    pf8 = malariagen_data.Pf8()
    variant_data = pf8.variant_calls()

    pos_var    = variant_data["variant_position"]
    chrom_var  = variant_data["variant_chrom"]
    chunk_sizes = pos_var.chunks[0]

    chunk_index = []
    offset  = 0
    n_chunks = len(chunk_sizes)
    progress = st.progress(0, text="Processing chunks...")

    for i, chunk_size in enumerate(chunk_sizes):
        chunk_start = offset
        chunk_end   = offset + chunk_size

        chunk_pos   = pos_var.isel(variants=slice(chunk_start, chunk_end)).values
        chunk_chrom = chrom_var.isel(variants=slice(chunk_start, chunk_end)).values

        for chrom in np.unique(chunk_chrom):
            chrom_mask     = chunk_chrom == chrom
            chrom_positions = chunk_pos[chrom_mask]
            chrom_indices  = np.where(chrom_mask)[0]

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


# ── Locus parsing & resolution ────────────────────────────────────────────────

def parse_loci_from_input(user_input: str) -> pd.DataFrame:
    rows = []
    for part in user_input.split():
        if "[" not in part or "]" not in part:
            continue

        identifier, pos_part = part.split("[", 1)
        pos_part = pos_part.rstrip("]")

        is_gene    = bool(re.match(r"^PF3D7_\d{7}$", identifier, re.IGNORECASE))
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


def _aa_to_genomic_intervals(gene_id: str, aa_start: int, aa_end: int,
                              exon_gff: pd.DataFrame) -> list[tuple]:
    """Map a single AA range to a list of (chrom, genomic_start, genomic_end) intervals."""
    exons = exon_gff[exon_gff["gene_id"] == gene_id].copy()
    if exons.empty:
        return []

    chrom  = exons["seqid"].iloc[0]
    strand = exons["strand"].iloc[0]
    # Sort exons in CDS order (ascending for +, descending for -)
    exons  = exons.sort_values("start", ascending=(strand == "+"))

    # Build CDS-coordinate segments
    cds_offset = 1
    segments   = []
    for _, exon in exons.iterrows():
        exon_len = int(exon["end"]) - int(exon["start"]) + 1
        segments.append({
            "cds_start":     cds_offset,
            "cds_end":       cds_offset + exon_len - 1,
            "genomic_start": int(exon["start"]),
            "genomic_end":   int(exon["end"]),
        })
        cds_offset += exon_len

    nt_start = (aa_start - 1) * 3 + 1
    nt_end   = aa_end * 3

    intervals = []
    for seg in segments:
        ov_start = max(nt_start, seg["cds_start"])
        ov_end   = min(nt_end,   seg["cds_end"])
        if ov_start > ov_end:
            continue

        offset_start = ov_start - seg["cds_start"]
        offset_end   = ov_end   - seg["cds_start"]

        if strand == "+":
            g_start = seg["genomic_start"] + offset_start
            g_end   = seg["genomic_start"] + offset_end
        else:
            g_end   = seg["genomic_end"] - offset_start
            g_start = seg["genomic_end"] - offset_end

        intervals.append((chrom, g_start, g_end))

    return intervals


def resolve_loci(loci_df: pd.DataFrame, exon_gff: pd.DataFrame) -> dict:
    """
    Group loci by source identifier and resolve to genomic (NT) intervals.

    Returns:
        {
            source_id: {
                "coord_type": "aa" | "nt",
                "intervals":  [(chrom, start, end), ...]
            }
        }
    """
    resolved = {}

    for source_id, group in loci_df.groupby("chrom", sort=False):
        coord_type = group["coord_type"].iloc[0]
        intervals  = []

        for _, row in group.iterrows():
            if coord_type == "nt":
                intervals.append((source_id, int(row["start"]), int(row["end"])))
            else:
                aa_intervals = _aa_to_genomic_intervals(
                    source_id, int(row["start"]), int(row["end"]), exon_gff
                )
                if not aa_intervals:
                    st.warning(f"Could not resolve AA positions {row['start']} - {row['end']} "
                               f"for `{source_id}`")
                intervals.extend(aa_intervals)

        resolved[source_id] = {"coord_type": coord_type, "intervals": intervals}

    return resolved


# ── Variant querying ──────────────────────────────────────────────────────────

def query_locus_metadata(_variant_data, _chunk_index_df,
                         source_id: str, locus_intervals: tuple) -> dict | None:
    """
    Query variant metadata for a set of genomic intervals.

    Args:
        locus_intervals: tuple of (chrom, start, end) - must be a tuple for cache hashing.
    """
    interval_datasets = []

    for chrom, start_pos, end_pos in locus_intervals:
        matching = _chunk_index_df[
            (_chunk_index_df["chrom"]   == chrom) &
            (_chunk_index_df["max_pos"] >= start_pos) &
            (_chunk_index_df["min_pos"] <= end_pos)
        ]
        if matching.empty:
            continue

        seen   = set()
        chunks = []
        for _, row in matching.iterrows():
            key = (int(row["chunk_start"]), int(row["chunk_end"]))
            if key in seen:
                continue
            seen.add(key)
            chunks.append(_variant_data.isel(variants=slice(*key)))

        if not chunks:
            continue

        region = xr.concat(chunks, dim="variants") if len(chunks) > 1 else chunks[0]
        mask = (
            (region["variant_chrom"].values    == chrom) &
            (region["variant_position"].values >= start_pos) &
            (region["variant_position"].values <= end_pos)
        )
        if mask.any():
            interval_datasets.append(region.isel(variants=mask))

    if not interval_datasets:
        return None

    result = (
        xr.concat(interval_datasets, dim="variants")
        if len(interval_datasets) > 1
        else interval_datasets[0]
    )

    alleles = result["variant_allele"].values
    if alleles.dtype.kind == "S":
        alleles = alleles.astype("U1")

    region_meta = {
        "positions":   result["variant_position"].values,
        "alleles":     alleles,
        "sample_ids":  result["sample_id"].values,
        "is_snp":      result["variant_is_snp"].values,
        "filter_pass": result["variant_filter_pass"].values,
        "CDS":         result["variant_CDS"].values,
        "numalt":      result["variant_numalt"].values,
        "n_variants":  result.dims["variants"],
        "n_samples":   result.dims["samples"],
    }

    # Stash xarray dataset in session state for deferred genotype loading
    st.session_state[f"_region_xr_{source_id}"] = result

    return region_meta


def load_genotypes(source_id: str) -> np.ndarray:
    return (
        st.session_state[f"_region_xr_{source_id}"]["call_genotype"]
        .values
        .astype(np.int8)
    )


# ── Haplotype helpers ─────────────────────────────────────────────────────────

def get_sample_genotypes(region_meta: dict, genotypes: np.ndarray,
                         sample_idx: int) -> pd.DataFrame:
    positions  = region_meta["positions"]
    alleles    = region_meta["alleles"]
    gt         = genotypes[:, sample_idx, :]
    n_variants = len(positions)
    gt1, gt2   = gt[:, 0], gt[:, 1]
    row_idx    = np.arange(n_variants)
    valid1, valid2 = gt1 >= 0, gt2 >= 0

    allele_1 = np.full(n_variants, ".", dtype="U1")
    allele_2 = np.full(n_variants, ".", dtype="U1")
    allele_1[valid1] = alleles[row_idx[valid1], gt1[valid1]]
    allele_2[valid2] = alleles[row_idx[valid2], gt2[valid2]]

    return pd.DataFrame({
        "position": positions,
        "allele_1": allele_1,
        "allele_2": allele_2,
    })


def get_haplotype_counts(region_meta: dict, genotypes: np.ndarray) -> pd.DataFrame:
    alleles    = region_meta["alleles"]
    n_variants, n_samples, _ = genotypes.shape

    gt1 = genotypes[:, :, 0]
    gt2 = genotypes[:, :, 1]
    row_idx = np.arange(n_variants)[:, np.newaxis]

    alleles_bytes = alleles.astype("S1") if alleles.dtype.kind == "U" else alleles
    gt1_c = np.clip(gt1, 0, alleles.shape[1] - 1)
    gt2_c = np.clip(gt2, 0, alleles.shape[1] - 1)

    a1 = np.where(gt1 >= 0, alleles_bytes[row_idx, gt1_c], b".")
    a2 = np.where(gt2 >= 0, alleles_bytes[row_idx, gt2_c], b".")

    hap1 = ["-".join(r) for r in a1.T.astype("U1")]
    hap2 = ["-".join(r) for r in a2.T.astype("U1")]
    all_haplotypes = hap1 + hap2

    counts = pd.Series(all_haplotypes).value_counts().reset_index()
    counts.columns = ["haplotype", "count"]
    counts["count"]     = counts["count"] // 2
    counts["frequency"] = counts["count"] / len(all_haplotypes) * 2

    return counts