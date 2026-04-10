import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import malariagen_data
import re
from scipy.stats import binom as _binom_dist


# ── Phasing-safety constants ──────────────────────────────────────────────────
# Heterozygous genotype calls at a position are classified as "safe to phase"
# only when all four criteria below are met.  A binomial one-sided test is used
# to ask: could the minor-allele reads simply be sequencing noise?
#
# _P_NOISE — noise baseline for the binomial null hypothesis.
#   Illumina instruments typically achieve per-base error rates of 0.1–1 %
#   (Phred Q20–Q30).  Setting p_noise = 0.02 (2 %) is deliberately conservative:
#   it sits above the empirical error rate, making it harder to reject the null
#   and therefore reducing false-positive "safe" calls.  Using the true error rate
#   (~0.5 %) would make the test nearly trivially pass for any minor allele above
#   a handful of reads; 2 % provides a more robust guard.
#
# _ALPHA — significance threshold for the binomial p-value.
#   A standard α = 0.05 is appropriate for a single test, but this test is
#   applied independently to every (sample × position) pair in a cohort, which
#   can run to tens of thousands of comparisons.  Using α = 0.01 provides
#   informal protection against the multiple-testing burden without requiring
#   full Bonferroni correction (which would demand α ~ 10⁻⁶ for large datasets
#   and would be excessively conservative for a per-call quality annotation).
#
# _MIN_TOTAL_AD — minimum total allele depth.
#   Allele-frequency confidence intervals widen rapidly below ~20× coverage.
#   The 95 % Clopper–Pearson interval for a minor VAF of 20 % at depth 20 is
#   approximately [0.06, 0.44] — already spanning a fourfold range.  Below 20×,
#   the binomial p-value can appear significant by chance even when the data are
#   uninformative (e.g. 3/3 reads at the minority allele).  This floor ensures
#   the test is only applied where depth is sufficient to meaningfully distinguish
#   single-strain from multi-strain genotypes.  A threshold of 20× is consistent
#   with widely adopted variant-calling best practices (e.g. GATK hard filters).
#
# _MIN_MINOR_AD — minimum absolute minor-allele read count.
#   Even at adequate total depth, very few supporting reads for the minor allele
#   are a red flag: a single miscalled base in an otherwise homozygous region
#   could yield 1–4 alt reads.  Requiring ≥ 5 minor-allele reads provides an
#   absolute floor that the binomial p-value alone cannot enforce, particularly
#   in regions with high reference bias or soft-clipping artefacts.
#
# _MAX_MINOR_VAF — maximum minor allele fraction for reliable ordering.
#   Even a genuine heterozygous call is labelled "unsafe" if the two alleles are
#   too close in frequency to order reliably.  If the true minor-allele fraction
#   is p, the probability of observing more minor reads than major reads in a
#   sample of n reads is P(Binomial(n, p) > n/2).  At p = 0.40 and n = 30 this
#   ordering-error probability is ~15 %; at p = 0.45 it exceeds 35 %.  A cap of
#   0.40 therefore requires at least a 60:40 split, keeping the probability of
#   mis-ordering the alleles below ~15 % even at the minimum eligible depth,
#   and well below 5 % at ≥ 50× coverage.  Calls above this threshold are still
#   reported as het (the call itself is valid) but the major/minor labelling
#   should not be used for haplotype inference.
_P_NOISE      = 0.02   # conservative noise baseline (above typical Illumina Q20–Q30 error rate)
_ALPHA        = 0.01   # per-test threshold; guards against multiple-testing inflation
_MIN_TOTAL_AD = 20     # depth floor; below this, allele-frequency CIs are too wide
_MIN_MINOR_AD = 5      # absolute read floor; prevents artefactual single-base miscalls
_MAX_MINOR_VAF = 0.40  # ordering reliability cap; >40% minor VAF → major/minor assignment unreliable


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
    return malariagen_data.Pf9().variant_calls()


# ── Chunk index ───────────────────────────────────────────────────────────────

_CHUNK_INDEX_PATH = "assets/chunk_index.csv"


@st.cache_data(show_spinner="Building chunk index from zarr...", persist="disk")
def build_chunk_index() -> pd.DataFrame:
    import os
    if os.path.exists(_CHUNK_INDEX_PATH):
        return pd.read_csv(_CHUNK_INDEX_PATH)

    variant_data = malariagen_data.Pf9().variant_calls()

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


def _compute_phasing_status(
    het: np.ndarray,
    missing: np.ndarray,
    ad_g1_vi: np.ndarray,
    ad_g2_vi: np.ndarray,
    g1_wins_vi: np.ndarray,
) -> np.ndarray:
    """Per-sample phasing status for one variant position.

    Returns a 1-D object array (n_samples) with one of:
        "-"          missing genotype
        "hom"        homozygous call
        "het:safe"   het; minor allele significantly above noise (binomial p < _ALPHA)
        "het:unsafe" het; minor allele may be noise

    The binomial test is one-sided (H0: minor-allele rate == _P_NOISE) and is
    applied vectorised over all het samples at once.
    """
    n = len(het)
    result = np.full(n, "hom", dtype=object)
    result[missing] = "-"

    if not het.any():
        return result

    total_ad = ad_g1_vi + ad_g2_vi
    minor_ad = np.where(g1_wins_vi, ad_g2_vi, ad_g1_vi)  # g1_wins → g1 is major

    het_total = total_ad[het]
    het_minor = minor_ad[het]

    # P(X >= k | n, p_noise) = binom.sf(k-1, n, p_noise)  (one-sided greater)
    pvals = _binom_dist.sf(het_minor - 1, het_total, _P_NOISE)

    safe = (
        (het_total >= _MIN_TOTAL_AD) &
        (het_minor >= _MIN_MINOR_AD) &
        (pvals < _ALPHA) &
        (het_minor / np.maximum(het_total, 1) <= _MAX_MINOR_VAF)
    )

    result[het] = np.where(safe, "het:safe", "het:unsafe")
    return result


_SAMPLE_CHUNK_SIZE = 100  # chunk size for in-memory dedup loop


@st.cache_data
def load_call_data(
    _ds: xr.Dataset, cache_key: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load genotypes and allele-depth data for all samples at once.

    Loading all samples in one call lets zarr fetch all sample-chunks in parallel
    (concurrent HTTP requests on GCS-backed stores).  Splitting into per-chunk
    requests serialises that I/O and is significantly slower.

    Args:
        _ds:       xr.Dataset for the region (prefixed with _ to exclude from cache key).
        cache_key: stable hashable key — pass (source_id, apply_filter_pass,
                   apply_numalt1, pos_tuple) from the caller.

    Returns:
        genotypes: (n_variants, n_samples, 2) int8
        g1_wins:   (n_variants, n_samples) bool
        ad_g1:     (n_variants, n_samples) int
        ad_g2:     (n_variants, n_samples) int
    """
    genotypes = _ds["call_genotype"].values.astype(np.int8)
    ad        = _ds["call_AD"].values          # (n_variants, n_samples, n_alleles)

    n_v, n_s  = genotypes.shape[:2]
    n_alleles = ad.shape[2]
    vi        = np.arange(n_v)[:, None]
    si        = np.arange(n_s)[None, :]
    g1 = np.minimum(genotypes[:, :, 0].clip(0), n_alleles - 1)
    g2 = np.minimum(genotypes[:, :, 1].clip(0), n_alleles - 1)

    ad_g1   = ad[vi, si, g1]
    ad_g2   = ad[vi, si, g2]
    g1_wins = ad_g1 >= ad_g2
    return genotypes, g1_wins, ad_g1, ad_g2


def build_allele_matrix(
    regions: dict,
    excluded_positions: set[str] | None = None,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Build a deduplicated allele table.

    Data must already be loaded into ``region["genotypes"]`` etc. (via
    ``load_call_data``).  Samples are processed in Python chunks of
    ``_SAMPLE_CHUNK_SIZE`` so that the full allele matrix is never materialised;
    unique allele combinations are folded into a dict as each chunk is processed.
    The result is the deduped format that ``compute_haplotypes`` expects
    (position columns + ``n_samples`` + ``sample_ids``), with no separate
    deduplication pass required.

    Het calls encode phasing confidence in the separator:
      - ``"major/minor"``  safe het  (minor allele significantly above noise)
      - ``"major|minor"``  unsafe het (low depth or VAF too close to 0.5)
    Missing calls are encoded as ``"-"``.

    Args:
        regions:            dict mapping source_id → region dict with pre-loaded
                            ``genotypes``, ``g1_wins``, ``ad_g1``, ``ad_g2``.
        excluded_positions: set of position strings to omit entirely.
        progress_cb:        optional callable(fraction: float, text: str) called
                            after each sample chunk (~540 calls for 53 k samples).

    Returns:
        DataFrame — one row per unique allele combination, with position columns
        plus ``n_samples`` (int) and ``sample_ids`` (list[str]).
    """
    excluded_positions = excluded_positions or set()

    # ── Phase 1: collect per-variant metadata and pre-compute call arrays ─────
    # Build the full per-variant arrays once (they are small — n_variants is tiny).
    # Sample iteration happens in Phase 2 over these arrays.
    col_labels: list[str]  = []
    seen_positions: set[str] = set()
    sample_ids = None

    # per-column arrays accumulated once across all regions
    call_arrays: list[np.ndarray] = []   # each (n_samples,) object

    for region in regions.values():
        meta      = region["meta"]
        genotypes = region["genotypes"]    # (n_variants, n_samples, 2)
        g1_wins   = region["g1_wins"]
        ad_g1     = region["ad_g1"]
        ad_g2     = region["ad_g2"]
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

            g1w     = g1_wins[vi]
            major   = np.where(g1w, a1, a2)
            minor   = np.where(g1w, a2, a1)
            # Phasing confidence is encoded in the allele separator rather than a separate column.
            # This keeps the information co-located with the call and avoids wide per-position
            # phasing columns in the output TSV.
            #
            # Separator convention:
            #   "major/minor"  safe het  — minor allele passes binomial test (p < _ALPHA),
            #                              depth ≥ _MIN_TOTAL_AD, minor reads ≥ _MIN_MINOR_AD,
            #                              and minor VAF ≤ _MAX_MINOR_VAF.  Major/minor ordering
            #                              by AD is reliable.
            #   "major|minor"  unsafe het — fails at least one criterion above.  The call itself
            #                              is still het; major/minor ordering may be unreliable.
            #
            # Both are handled symmetrically downstream (split on "/" or "|"); the separator
            # is preserved through _add_aa_haplotypes / _add_nt_haplotypes into ns_changes
            # notation (e.g. "K76[I/K]" vs "K76[I|K]").
            #
            # "|" was chosen over other options because:
            #   - GATK uses "|" for phased GTs in VCF (0|1), but that is in integer allele
            #     index space.  In amino-acid / nucleotide string space the ambiguity is absent.
            #   - "/" is already the natural "or" separator for safe ordered hets.
            #   - "*" and "!" are reserved for spanning deletions and stop codons respectively.
            phasing_status = _compute_phasing_status(het, missing, ad_g1[vi], ad_g2[vi], g1w)
            seps    = np.where(phasing_status == "het:safe", "/", "|")
            ordered = np.array([f"{m}{s}{n}" for m, s, n in zip(major, seps, minor)], dtype=object)
            calls   = np.where(missing, "-", np.where(het, ordered, a1))

            seen_positions.add(label)
            col_labels.append(label)
            call_arrays.append(np.asarray(calls, dtype=object))

    n_positions = len(col_labels)

    if n_positions == 0 or sample_ids is None:
        return pd.DataFrame(columns=["n_samples", "sample_ids"])

    n_samples = len(sample_ids)
    # Stack: (n_positions, n_samples) — column-major so row-slicing is a simple index
    calls_matrix = np.stack(call_arrays, axis=0)   # (n_pos, n_samples)

    # ── Phase 2: iterate over sample chunks, fold into dedup dict ─────────────
    dedup: dict[tuple, dict] = {}
    n_chunks = (n_samples + _SAMPLE_CHUNK_SIZE - 1) // _SAMPLE_CHUNK_SIZE

    for chunk_i in range(n_chunks):
        chunk_start = chunk_i * _SAMPLE_CHUNK_SIZE
        chunk_end   = min(chunk_start + _SAMPLE_CHUNK_SIZE, n_samples)
        chunk_ids   = sample_ids[chunk_start:chunk_end]

        # Slice already-loaded arrays — no I/O
        chunk_calls = calls_matrix[:, chunk_start:chunk_end]    # (n_pos, chunk_size)

        chunk_size = chunk_end - chunk_start
        for s in range(chunk_size):
            key = tuple(chunk_calls[:, s])
            if key in dedup:
                dedup[key]["n"] += 1
                dedup[key]["ids"].append(str(chunk_ids[s]))
            else:
                dedup[key] = {"n": 1, "ids": [str(chunk_ids[s])]}

        if progress_cb is not None:
            progress_cb(
                (chunk_i + 1) / n_chunks,
                f"Deduplicating samples {chunk_end:,} / {n_samples:,}…",
            )

    # ── Phase 3: assemble output ──────────────────────────────────────────────
    rows = [
        {**dict(zip(col_labels, key)), "n_samples": info["n"], "sample_ids": info["ids"]}
        for key, info in dedup.items()
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=col_labels + ["n_samples", "sample_ids"]
    )
