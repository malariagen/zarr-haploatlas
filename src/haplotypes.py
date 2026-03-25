import streamlit as st
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


REF_FASTA = "assets/PlasmoDB-54_Pfalciparum3D7_Genome.fasta"


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_allele_matrix(allele_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse allele_matrix (samples × positions) to unique allele combinations.

    Returns a DataFrame with the same position columns plus:
      - n_samples : number of samples with this combination
      - sample_ids: list of their IDs
    """
    pos_cols = list(allele_matrix.columns)
    return (
        allele_matrix
        .reset_index(names="sample_id")
        .groupby(pos_cols, sort=False)["sample_id"]
        .agg(n_samples="count", sample_ids=list)
        .reset_index()
    )


# ── Reference helpers ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading reference genome…")
def load_ref_genome(fasta_path: str = REF_FASTA) -> dict[str, str]:
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}


def _build_ref_cds(gene_id: str, cds_gff: pd.DataFrame,
                   ref_genome: dict) -> tuple[str, str] | None:
    """Return (cds_sequence, strand) for a gene, or None if not found."""
    exons = cds_gff[cds_gff["gene_id"] == gene_id].copy()
    if exons.empty:
        return None

    chrom  = exons["seqid"].iloc[0]
    strand = exons["strand"].iloc[0]
    exons  = exons.sort_values("start", ascending=(strand == "+"))

    if chrom not in ref_genome:
        return None

    # GFF is 1-based inclusive; Python slicing is 0-based exclusive
    cds_parts = [ref_genome[chrom][int(e["start"]) - 1 : int(e["end"])] for _, e in exons.iterrows()]
    return "".join(cds_parts), strand


def _genomic_to_cds_offset(genomic_pos: int, exons_sorted: pd.DataFrame,
                            strand: str) -> int | None:
    """Map a 1-based genomic position to a 0-based CDS offset, or None if outside all exons."""
    cds_offset = 0
    for _, exon in exons_sorted.iterrows():
        exon_start = int(exon["start"])
        exon_end   = int(exon["end"])
        if exon_start <= genomic_pos <= exon_end:
            within = (genomic_pos - exon_start) if strand == "+" else (exon_end - genomic_pos)
            return cds_offset + within
        cds_offset += exon_end - exon_start + 1
    return None


def _translate(cds_seq: str, strand: str) -> str:
    if len(cds_seq) % 3 != 0:
        return "!"
    seq = Seq(cds_seq)
    if strand == "-":
        seq = seq.reverse_complement()
    return str(seq.translate(to_stop=True))


def _apply_variants(ref_seq: str, sorted_pos_info: list[tuple],
                    alleles_here: dict) -> str:
    """
    Apply variants (SNPs and indels) to ref_seq with running-offset tracking.

    sorted_pos_info: list of (pos_str, {"ref": str, "offset": int}) sorted by offset ascending.
    alleles_here:    {pos_str: observed_allele}

    Spanning deletions (*) and het (~) are treated as reference.
    Indels shift subsequent positions via a running offset (same logic as older.py).
    """
    seq_chars      = list(ref_seq)
    running_offset = 0

    for pos_str, info in sorted_pos_info:
        observed = alleles_here.get(pos_str, info["ref"])
        ref_a    = info["ref"]
        off      = info["offset"] + running_offset

        if observed in (ref_a, "*", "~"):
            continue

        ref_len = len(ref_a)
        alt_len = len(observed)
        seq_chars[off : off + ref_len] = list(observed)
        running_offset += alt_len - ref_len

    return "".join(seq_chars)


# ── Haplotype computation ─────────────────────────────────────────────────────

def compute_haplotypes(deduped: pd.DataFrame, regions: dict, resolved: dict,
                       loci_df: pd.DataFrame, cds_gff: pd.DataFrame,
                       ref_fasta_path: str = REF_FASTA) -> pd.DataFrame:
    """
    Compute haplotypes for each unique allele combination in `deduped`.

    For AA loci, output covers only the user-queried AA positions (e.g. "SVMNK,F").
    For NT loci, output covers each queried genomic interval independently (e.g. "G,ATTGTT").
    Intervals within a locus are comma-separated in the output.

    For each AA locus adds:
      - {source_id}_haplotype  : queried AA sub-sequences joined by ","
      - {source_id}_ns_changes : AA changes at queried positions vs reference (e.g. "C72S/I74M")

    For each NT locus adds:
      - {source_id}_haplotype  : queried NT sequences joined by ","

    Variant handling:
      - SNPs and indels   : applied with running-offset tracking
      - Spanning del (*)  : treated as reference
      - Het (~)           : treated as reference; locus flagged with "~" suffix
      - Missing (-)       : whole locus result set to "-"
    """
    ref_genome = load_ref_genome(ref_fasta_path)
    result     = deduped.copy()

    for source_id, locus_info in resolved.items():
        region = regions.get(source_id)
        if region is None:
            continue

        query_ranges = loci_df[loci_df["chrom"] == source_id][["start", "end"]].values.tolist()

        if locus_info["coord_type"] == "aa":
            _add_aa_haplotypes(result, deduped, source_id, region["meta"],
                               query_ranges, cds_gff, ref_genome)
        else:
            _add_nt_haplotypes(result, deduped, source_id, region["meta"],
                               query_ranges, locus_info["intervals"], ref_genome)

    return result


# ── AA loci ───────────────────────────────────────────────────────────────────

def _add_aa_haplotypes(result, deduped, source_id, meta, aa_ranges,
                       cds_gff, ref_genome):
    cds_result = _build_ref_cds(source_id, cds_gff, ref_genome)
    if cds_result is None:
        return
    ref_cds, strand = cds_result
    ref_aa = _translate(ref_cds, strand)

    exons = cds_gff[cds_gff["gene_id"] == source_id].copy()
    exons = exons.sort_values("start", ascending=(strand == "+"))

    # Map queried variant positions to 0-based CDS offsets
    pos_info = {}
    for vi, pos in enumerate(meta["positions"]):
        cds_off = _genomic_to_cds_offset(int(pos), exons, strand)
        if cds_off is None:
            continue
        pos_info[str(pos)] = {"ref": str(meta["alleles"][vi][0]), "offset": cds_off}

    sorted_pos_info = sorted(pos_info.items(), key=lambda x: x[1]["offset"])

    haplotypes     = []
    ns_changes_col = []

    for _, row in deduped.iterrows():
        alleles_here = {p: row[p] for p in pos_info if p in deduped.columns}
        has_missing  = any(v == "-" for v in alleles_here.values())
        has_het      = any(v == "~" for v in alleles_here.values())

        if has_missing:
            haplotypes.append("-")
            ns_changes_col.append("-")
            continue

        alt_cds = _apply_variants(ref_cds, sorted_pos_info, alleles_here)
        alt_aa  = _translate(alt_cds, strand)
        suffix  = "~" if has_het else ""

        hap_parts = []
        ns_parts  = []
        for aa_start, aa_end in aa_ranges:
            ref_slice = ref_aa[aa_start - 1 : aa_end] if len(ref_aa) >= aa_end else ""
            alt_slice = alt_aa[aa_start - 1 : aa_end] if len(alt_aa) >= aa_end else ""

            if not ref_slice or len(ref_slice) != len(alt_slice):
                hap_parts.append("!")
                ns_parts.append("!")
                continue

            hap_parts.append(alt_slice)
            changes = [
                f"{ref_slice[i]}{aa_start + i}{alt_slice[i]}"
                for i in range(len(ref_slice))
                if ref_slice[i] != alt_slice[i]
            ]
            ns_parts.append("/".join(changes))

        haplotypes.append(",".join(hap_parts) + suffix)
        ns_changes_col.append(",".join(ns_parts) + suffix)

    result[f"{source_id}_haplotype"]  = haplotypes
    result[f"{source_id}_ns_changes"] = ns_changes_col


# ── NT loci ───────────────────────────────────────────────────────────────────

def _add_nt_haplotypes(result, deduped, source_id, meta, nt_ranges,
                       intervals, ref_genome):
    # Build per-interval pos_info with offsets local to each interval
    interval_pos_infos = []   # one entry per interval: sorted list of (pos_str, {ref, offset})

    for chrom, iv_start, iv_end in intervals:
        if chrom not in ref_genome:
            interval_pos_infos.append([])
            continue

        local_pos_info = {}
        for vi, pos in enumerate(meta["positions"]):
            gpos = int(pos)
            if iv_start <= gpos <= iv_end:
                local_pos_info[str(pos)] = {
                    "ref":    str(meta["alleles"][vi][0]),
                    "offset": gpos - iv_start,   # 0-based offset within this interval
                }

        interval_pos_infos.append(
            sorted(local_pos_info.items(), key=lambda x: x[1]["offset"])
        )

    haplotypes = []

    for _, row in deduped.iterrows():
        all_pos = {p for infos in interval_pos_infos for p, _ in infos}
        alleles_here = {p: row[p] for p in all_pos if p in deduped.columns}
        has_missing  = any(v == "-" for v in alleles_here.values())
        has_het      = any(v == "~" for v in alleles_here.values())

        if has_missing:
            haplotypes.append("-")
            continue

        parts = []
        for (chrom, iv_start, iv_end), sorted_pos_info in zip(intervals, interval_pos_infos):
            ref_iv  = ref_genome[chrom][iv_start - 1 : iv_end]
            alt_seq = _apply_variants(ref_iv, sorted_pos_info, alleles_here)
            parts.append(alt_seq)

        suffix = "~" if has_het else ""
        haplotypes.append(",".join(parts) + suffix)

    result[f"{source_id}_haplotype"] = haplotypes
