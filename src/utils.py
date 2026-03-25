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
        "cds_gff": gff[gff["type"] == "CDS"].reset_index(drop=True),
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
            chrom_mask      = chunk_chrom == chrom
            chrom_positions = chunk_pos[chrom_mask]
            chrom_indices   = np.where(chrom_mask)[0]

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
                              cds_gff: pd.DataFrame) -> list[tuple]:
    """Map a single AA range to a list of (chrom, genomic_start, genomic_end) intervals."""
    exons = cds_gff[cds_gff["gene_id"] == gene_id].copy()
    if exons.empty:
        return []

    chrom  = exons["seqid"].iloc[0]
    strand = exons["strand"].iloc[0]
    exons  = exons.sort_values("start", ascending=(strand == "+"))

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


def resolve_loci(loci_df: pd.DataFrame, cds_gff: pd.DataFrame) -> dict:
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
                    source_id, int(row["start"]), int(row["end"]), cds_gff
                )
                if not aa_intervals:
                    st.warning(f"Could not resolve AA positions {row['start']} - {row['end']} "
                               f"for `{source_id}`")
                intervals.extend(aa_intervals)

        resolved[source_id] = {"coord_type": coord_type, "intervals": intervals}

    return resolved


# ── Variant querying ──────────────────────────────────────────────────────────

def query_locus_metadata(_variant_data, _chunk_index_df,
                         locus_intervals: tuple) -> tuple[dict, xr.Dataset] | None:
    """
    Query variant metadata for a set of genomic intervals.

    Args:
        locus_intervals: tuple of (chrom, start, end) — must be a tuple for cache hashing.

    Returns:
        (meta dict, xr.Dataset) or None if no variants found.
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

    ds = (
        xr.concat(interval_datasets, dim="variants")
        if len(interval_datasets) > 1
        else interval_datasets[0]
    )

    alleles = ds["variant_allele"].values
    if alleles.dtype.kind == "S":
        alleles = alleles.astype("U1")

    meta = {
        "positions":   ds["variant_position"].values,
        "alleles":     alleles,
        "sample_ids":  ds["sample_id"].values,
        "is_snp":      ds["variant_is_snp"].values,
        "filter_pass": ds["variant_filter_pass"].values,
        "CDS":         ds["variant_CDS"].values,
        "numalt":      ds["variant_numalt"].values,
        "n_variants":  ds.dims["variants"],
        "n_samples":   ds.dims["samples"],
    }

    return meta, ds


def filter_region_by_intervals(meta: dict, ds: xr.Dataset,
                                intervals: list[tuple]) -> tuple[dict, xr.Dataset] | None:
    """
    Post-filter a queried region to variants whose REF footprint overlaps at least
    one of the given intervals. Handles long REF alleles from upstream indels.
    """
    positions = meta["positions"]
    alleles   = meta["alleles"]
    keep_mask = np.zeros(len(positions), dtype=bool)

    for vi in range(len(positions)):
        vpos    = int(positions[vi])
        v_end   = vpos + len(str(alleles[vi][0])) - 1
        for _, iv_start, iv_end in intervals:
            if vpos <= iv_end and v_end >= iv_start:
                keep_mask[vi] = True
                break

    if not keep_mask.any():
        return None

    keep_idx      = np.where(keep_mask)[0]
    filtered_meta = {
        "positions":   positions[keep_idx],
        "alleles":     alleles[keep_idx],
        "sample_ids":  meta["sample_ids"],
        "is_snp":      meta["is_snp"][keep_idx],
        "filter_pass": meta["filter_pass"][keep_idx],
        "CDS":         meta["CDS"][keep_idx],
        "numalt":      meta["numalt"][keep_idx],
        "n_variants":  int(keep_idx.shape[0]),
        "n_samples":   meta["n_samples"],
    }
    return filtered_meta, ds.isel(variants=keep_idx)


def build_variant_rows(source_id: str, meta: dict, locus_info: dict) -> list[dict]:
    """Build display rows for the variant metadata table."""
    coord_type  = locus_info["coord_type"]
    locus_chrom = locus_info["intervals"][0][0] if locus_info["intervals"] else ""
    rows = []
    for vi in range(meta["n_variants"]):
        ref_allele = str(meta["alleles"][vi][0])
        alts = [str(a) for a in meta["alleles"][vi][1:] if str(a).strip()]
        rows.append({
            "chrom":             locus_chrom,
            "position":          meta["positions"][vi],
            "source":            source_id,
            "needs_translation": coord_type == "aa",
            "ref":               ref_allele,
            "ref_len":           len(ref_allele),
            "alt":               ", ".join(alts),
            "is_snp":            meta["is_snp"][vi],
            "filter_pass":       meta["filter_pass"][vi],
            "CDS":               meta["CDS"][vi],
            "numalt":            meta["numalt"][vi],
        })
    return rows


def load_genotypes(ds: xr.Dataset) -> np.ndarray:
    return ds["call_genotype"].values.astype(np.int8)


def build_allele_matrix(regions: dict) -> pd.DataFrame:
    """
    Build a samples × positions DataFrame of called alleles.

    Columns are labelled "chrom:pos". Cells show the allele string:
    homozygous → single allele, heterozygous → "A/T", missing → ".".
    """
    col_labels = []
    col_arrays = []

    sample_ids = None

    for region in regions.values():
        meta      = region["meta"]
        genotypes = region["genotypes"]   # (n_variants, n_samples, 2)
        positions = meta["positions"]

        if sample_ids is None:
            sample_ids = meta["sample_ids"]

        for vi in range(meta["n_variants"]):
            gt1 = genotypes[vi, :, 0]   # (n_samples,)
            gt2 = genotypes[vi, :, 1]

            allele_row = meta["alleles"][vi]
            n_alleles  = allele_row.shape[0]

            gt1c = np.clip(gt1, 0, n_alleles - 1)
            gt2c = np.clip(gt2, 0, n_alleles - 1)

            a1 = np.where(gt1 >= 0, np.array(allele_row)[gt1c], "")
            a2 = np.where(gt2 >= 0, np.array(allele_row)[gt2c], "")

            missing = (gt1 < 0) | (gt2 < 0)
            het     = ~missing & (a1 != a2)
            # "~" for het, "-" for missing; "*" is reserved for spanning deletions in VCF
            calls = np.where(
                missing, "-",
                np.where(het, "~", a1)
            )

            col_labels.append(f"{positions[vi]}")
            col_arrays.append(calls)

    return pd.DataFrame(
        np.column_stack(col_arrays) if col_arrays else np.empty((len(sample_ids), 0), dtype=str),
        index=sample_ids,
        columns=col_labels,
    )
