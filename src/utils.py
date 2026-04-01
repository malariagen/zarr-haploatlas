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


@st.cache_resource(show_spinner="Connecting to variant data…")
def load_variant_data():
    # cache_resource keeps a single reference; cache_data would pickle the entire
    # zarr-backed Dataset into a copy, consuming several hundred MB unnecessarily.
    return malariagen_data.Pf8().variant_calls()


# ── Chunk index ───────────────────────────────────────────────────────────────

_CHUNK_INDEX_PATH = "assets/chunk_index.csv"


@st.cache_data(show_spinner="Building chunk index from zarr...", persist="disk")
def build_chunk_index() -> pd.DataFrame:
    import os
    if os.path.exists(_CHUNK_INDEX_PATH):
        return pd.read_csv(_CHUNK_INDEX_PATH)

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
    df = pd.DataFrame(chunk_index)
    df.to_csv(_CHUNK_INDEX_PATH, index=False)
    return df


# ── Locus parsing & resolution ────────────────────────────────────────────────

def parse_loci_from_input(user_input: str) -> pd.DataFrame:
    # Match IDENTIFIER[...] tokens with optional (alias) suffix.
    tokens = re.findall(r'([^\s\[]+\[[^\]]+\](?:\([^)]+\))?)', user_input)
    rows = []
    for token in tokens:
        # Extract optional alias like (crt)
        alias_match = re.search(r'\(([^)]+)\)$', token)
        alias = alias_match.group(1) if alias_match else None
        token_body = token[:alias_match.start()].strip() if alias_match else token

        identifier, pos_part = token_body.split("[", 1)
        pos_part = pos_part.rstrip("]")

        is_gene    = bool(re.match(r"^PF3D7_(?:\d{7}|(?:MIT|API)\d{5})$", identifier, re.IGNORECASE))
        coord_type = "aa" if is_gene else "nt"

        for part in pos_part.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "*":
                # Wildcard: full gene — resolved later by expand_full_gene_loci
                rows.append((identifier, 0, 0, coord_type, alias))
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                rows.append((identifier, int(start.strip()), int(end.strip()), coord_type, alias))
            else:
                pos = int(part)
                rows.append((identifier, pos, pos, coord_type, alias))

    return pd.DataFrame(rows, columns=["chrom", "start", "end", "coord_type", "alias"])


def _compute_gene_aa_length(gene_id: str, cds_gff: pd.DataFrame) -> int:
    """Return the protein length (AA count, excluding stop codon) for a gene.

    PlasmoDB CDS annotations include the stop codon in the exon coordinates,
    so we subtract one codon from the total to get the protein-coding length.
    """
    exons = cds_gff[cds_gff["gene_id"] == gene_id]
    if exons.empty:
        return 0
    total_nt = sum(int(row["end"]) - int(row["start"]) + 1 for _, row in exons.iterrows())
    n_codons = total_nt // 3
    return max(0, n_codons - 1)  # exclude stop codon


def expand_full_gene_loci(parsed_loci: pd.DataFrame, cds_gff: pd.DataFrame) -> pd.DataFrame:
    """
    Replace wildcard rows (start == end == 0, coord_type == 'aa') with the
    full AA range 1..n_aa by looking up the gene's CDS length in the GFF.
    Rows that cannot be resolved are silently dropped.
    """
    if parsed_loci.empty:
        return parsed_loci
    out = []
    for _, row in parsed_loci.iterrows():
        if row["coord_type"] == "aa" and row["start"] == 0 and row["end"] == 0:
            n_aa = _compute_gene_aa_length(str(row["chrom"]), cds_gff)
            if n_aa > 0:
                out.append({**row.to_dict(), "start": 1, "end": n_aa})
            # else: gene not in GFF — skip silently
        else:
            out.append(row.to_dict())
    if not out:
        return pd.DataFrame(columns=parsed_loci.columns)
    return pd.DataFrame(out, columns=parsed_loci.columns).reset_index(drop=True)


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


def resolve_loci(
    loci_df: pd.DataFrame,
    cds_gff: pd.DataFrame,
    verbose: bool = False,
) -> dict | tuple[dict, tuple[str, ...]]:
    """
    Resolve loci to genomic intervals.

    If verbose is True, print notices for unresolved AA ranges.
    """
    resolved = {}
    notices: list[str] = []

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
                    notices.append(
                        f"Could not resolve AA positions {row['start']} - {row['end']} for `{source_id}`"
                    )
                intervals.extend(aa_intervals)

        resolved[source_id] = {"coord_type": coord_type, "intervals": intervals}

    if verbose:
        for notice in notices:
            print(notice)
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
        alleles = alleles.astype("U")   # decode bytes → unicode, preserving full allele length
    # strip trailing '-' and null padding (zarr fixed-width allele convention)
    alleles = np.vectorize(lambda s: s.rstrip("-\x00"))(alleles)

    meta = {
        "positions":   ds["variant_position"].values,
        "alleles":     alleles,
        "sample_ids":  ds["sample_id"].values,
        "is_snp":      ds["variant_is_snp"].values,
        "filter_pass": ds["variant_filter_pass"].values,
        "CDS":         ds["variant_CDS"].values,
        "numalt":      ds["variant_numalt"].values,
        "n_variants":  ds.sizes["variants"],
        "n_samples":   ds.sizes["samples"],
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

@st.cache_data(show_spinner = False)
def build_regions(
    resolved_loci: dict,
    _variant_data,
    chunk_index_df: pd.DataFrame,
    upstream_pad: int = 100,
) -> tuple[dict, list[str]]:
    """
    Query variant data for each resolved locus.

    Returns:
        regions:  dict mapping source_id → {"meta", "ds", "genotypes", "allele_depths"}
        warnings: list of human-readable warning strings for failed/empty loci
    """
    regions: dict = {}
    warnings: list[str] = []

    for source_id, locus_info in resolved_loci.items():
        intervals = locus_info["intervals"]
        if not intervals:
            warnings.append(f"No genomic intervals resolved for `{source_id}`")
            continue

        padded = tuple((c, max(1, s - upstream_pad), e) for c, s, e in intervals)
        result = query_locus_metadata(_variant_data, chunk_index_df, padded)
        if result is None:
            warnings.append(f"No variants found for `{source_id}`")
            continue

        filtered = filter_region_by_intervals(*result, intervals)
        if filtered is None:
            warnings.append(f"No variants found for `{source_id}`")
            continue

        meta, ds = filtered
        regions[source_id] = {"meta": meta, "ds": ds, "genotypes": None, "g1_wins": None}

    return regions, warnings


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
            "alt":               alts,
            "is_snp":            meta["is_snp"][vi],
            "filter_pass":       meta["filter_pass"][vi],
            "CDS":               meta["CDS"][vi],
            "numalt":            meta["numalt"][vi],
        })
    return rows


@st.cache_data
def load_call_data(_ds: xr.Dataset, cache_key: tuple, load_ad: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Load genotypes and, if requested, a precomputed allele-depth comparison.

    Args:
        _ds:       xr.Dataset for the region. Prefixed with _ so Streamlit excludes it
                   from the cache key (xr.Dataset is not hashable).
        cache_key: stable hashable key that uniquely identifies this dataset —
                   pass (source_id, apply_filter_pass, apply_numalt1) from the caller.
                   Must change whenever _ds changes.
        load_ad:   set to False to skip AD entirely (e.g. when het_mode="exclude").
                   Halves the data downloaded from GCS.

    Returns:
        genotypes: (n_variants, n_samples, 2) int8
        g1_wins:   (n_variants, n_samples) bool — True where the first called allele
                   has >= depth than the second. None if load_ad=False.
    """
    genotypes = _ds["call_genotype"].values.astype(np.int8)

    if not load_ad:
        return genotypes, None

    ad = _ds["call_AD"].values  # (n_variants, n_samples, n_alleles)

    n_v, n_s   = genotypes.shape[:2]
    n_alleles  = ad.shape[2]
    vi         = np.arange(n_v)[:, None]
    si         = np.arange(n_s)[None, :]
    g1 = np.minimum(genotypes[:, :, 0].clip(0), n_alleles - 1)
    g2 = np.minimum(genotypes[:, :, 1].clip(0), n_alleles - 1)

    g1_wins = ad[vi, si, g1] >= ad[vi, si, g2]  # (n_variants, n_samples) bool
    return genotypes, g1_wins


def build_allele_matrix(
    regions: dict,
    excluded_positions: set[str] | None = None,
    het_mode: str = "exclude",
    het_sep: str = "/",
) -> pd.DataFrame:
    """
    Build a samples × positions DataFrame of called alleles.

    het_mode:
      "exclude"  – het → "#", missing → "-"
      "major_ad" – resolve het to the allele with higher depth (requires g1_wins)

    excluded_positions: set of position strings to omit entirely.
    """
    excluded_positions = excluded_positions or set()
    col_labels: list[str] = []
    col_arrays: list      = []
    seen_positions: set[str] = set()
    sample_ids = None

    for region in regions.values():
        meta     = region["meta"]
        genotypes = region["genotypes"]   # (n_variants, n_samples, 2)
        g1_wins   = region.get("g1_wins") # (n_variants, n_samples) bool, or None
        positions = meta["positions"]

        if sample_ids is None:
            sample_ids = meta["sample_ids"]

        for vi in range(meta["n_variants"]):
            label = str(positions[vi])
            if label in seen_positions or label in excluded_positions:
                continue

            gt1 = genotypes[vi, :, 0]
            gt2 = genotypes[vi, :, 1]
            allele_row = meta["alleles"][vi]
            n_alleles  = allele_row.shape[0]

            gt1c = np.clip(gt1, 0, n_alleles - 1)
            gt2c = np.clip(gt2, 0, n_alleles - 1)

            a1 = np.where(gt1 >= 0, np.array(allele_row)[gt1c], "")
            a2 = np.where(gt2 >= 0, np.array(allele_row)[gt2c], "")

            missing = (gt1 < 0) | (gt2 < 0)
            het     = ~missing & (a1 != a2)

            if het_mode == "major_ad" and g1_wins is not None:
                calls = np.where(missing, "-", np.where(het, np.where(g1_wins[vi], a1, a2), a1))
            elif het_mode == "ordered_ad" and g1_wins is not None:
                # "major/minor" ordered by allele depth
                major   = np.where(g1_wins[vi], a1, a2)
                minor   = np.where(g1_wins[vi], a2, a1)
                ordered = np.array([f"{m}{het_sep}{n}" for m, n in zip(major, minor)], dtype=object)
                calls   = np.where(missing, "-", np.where(het, ordered, a1))
            else:
                calls = np.where(missing, "-", np.where(het, "#", a1))

            seen_positions.add(label)
            col_labels.append(label)
            col_arrays.append(np.asarray(calls, dtype=object))

    if not col_arrays:
        return pd.DataFrame(
            np.empty((len(sample_ids), 0), dtype=object),
            index=sample_ids, columns=[],
        )
    return pd.DataFrame(
        np.column_stack(col_arrays),
        index=sample_ids,
        columns=col_labels,
    )
