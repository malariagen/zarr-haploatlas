import streamlit as st
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


REF_FASTA = "assets/PlasmoDB-54_Pfalciparum3D7_Genome.fasta"

HET_SYMBOL = "*"   # displayed in allele matrix and haplotype columns


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
    exons  = exons.sort_values("start", ascending=True)

    if chrom not in ref_genome:
        return None

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
            return cds_offset + (genomic_pos - exon_start)
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

    Spanning deletions (*), het (*), multi-allele cells ("G,T"), and missing ("-")
    are treated as reference. Indels shift subsequent positions via a running offset.
    """
    seq_chars      = list(ref_seq)
    running_offset = 0

    for pos_str, info in sorted_pos_info:
        observed = alleles_here.get(pos_str, info["ref"])
        ref_a    = info["ref"]

        # Multi-allele cells ("G,T") and ordered-ad pairs ("G/T") → treat as reference;
        # the callers that need both translations handle them before calling this function.
        if isinstance(observed, str) and ("," in observed or "/" in observed):
            continue

        # Strip trailing '-'/null padding; leave special markers untouched
        if observed not in ("*", "-"):
            observed = observed.rstrip("-\x00")
        ref_a = ref_a.rstrip("-\x00")

        off = info["offset"] + running_offset

        # "*" covers both het and VCF spanning deletion — treat as reference
        if observed in (ref_a, "*", "-"):
            continue

        ref_len = len(ref_a)
        alt_len = len(observed)
        seq_chars[off : off + ref_len] = list(observed)
        running_offset += alt_len - ref_len

    return "".join(seq_chars)


def _aa_at(alt_aa: str, pos: int) -> str:
    """Return amino acid at 1-based position, '!' if beyond sequence (premature stop)."""
    if pos < 1 or pos > len(alt_aa):
        return "!"
    return alt_aa[pos - 1]


def _aa_slice(alt_aa: str, start: int, end: int) -> str:
    """Return AA substring for 1-based inclusive [start, end], '!' if truncated."""
    if start < 1 or end > len(alt_aa):
        return "!"
    return alt_aa[start - 1 : end]


# ── Haplotype computation ─────────────────────────────────────────────────────

def compute_haplotypes(deduped: pd.DataFrame, regions: dict, resolved: dict,
                       loci_df: pd.DataFrame, cds_gff: pd.DataFrame,
                       ref_fasta_path: str = REF_FASTA,
                       het_sep: str = "/",
                       mode: str = "default") -> pd.DataFrame:
    """
    Compute haplotypes for each unique allele combination in `deduped`.

    For AA loci adds:
      - {source_id}_haplotype  : queried AA sub-sequences joined by ","
      - {source_id}_ns_changes : AA changes vs reference
      - {source_id}_{pos}      : per-queried-position AA column (single pos) or
        {source_id}_{start}_{end} (range)

    For NT loci adds:
      - {source_id}_haplotype  : queried NT sequences joined by ","
      - {source_id}_{start}_{end} (or {source_id}_{pos}): per-interval NT column

    Variant handling:
      - SNPs / indels          : applied with running-offset tracking
      - Spanning del (*) / het (*) : treated as reference; range flagged with "*"
      - Multi-allele ("G,T")   : treated as reference for full haplotype; per-position
                                  shows comma-separated translated AAs
      - Missing (-)            : whole locus / range result set to "-"
      - Stop codon (premature) : positions beyond the stop shown as "X"
    """
    ref_genome  = load_ref_genome(ref_fasta_path)
    result      = deduped.copy()
    new_col_dfs = []

    # Build alias map: source_id → display prefix
    alias_map: dict[str, str] = {}
    if "alias" in loci_df.columns:
        for chrom, grp in loci_df.groupby("chrom", sort=False):
            a = grp["alias"].dropna()
            if not a.empty:
                alias_map[chrom] = a.iloc[0]

    for source_id, locus_info in resolved.items():
        region = regions.get(source_id)
        if region is None:
            continue

        query_ranges = loci_df[loci_df["chrom"] == source_id][["start", "end"]].values.tolist()
        prefix = alias_map.get(source_id, source_id)

        if locus_info["coord_type"] == "aa":
            cols = _add_aa_haplotypes(deduped, source_id, region["meta"],
                                      query_ranges, cds_gff, ref_genome,
                                      het_sep=het_sep, prefix=prefix, mode=mode)
        else:
            cols = _add_nt_haplotypes(deduped, source_id, region["meta"],
                                      query_ranges, locus_info["intervals"], ref_genome,
                                      het_sep=het_sep, prefix=prefix)

        if cols:
            new_col_dfs.append(pd.DataFrame(cols, index=result.index))

    if new_col_dfs:
        result = pd.concat([result] + new_col_dfs, axis=1)

    return result


# ── AA loci ───────────────────────────────────────────────────────────────────

def _range_col_name(source_id: str, start: int, end: int) -> str:
    return f"{source_id}_{start}" if start == end else f"{source_id}_{start}_{end}"


def _pos_summary(ranges: list[tuple[int, int]]) -> str:
    """Compact position summary matching the output filename convention.

    [(72, 76), (220, 220), (271, 271)] → "72-76.220.271"
    """
    return ".".join(f"{s}-{e}" if s != e else str(s) for s, e in ranges)


_MAX_PER_POS_COLS = 50  # skip individual-position columns for very wide queries


def _add_aa_haplotypes(deduped, source_id, meta, aa_ranges,
                       cds_gff, ref_genome, het_sep: str = "/",
                       prefix: str | None = None, mode: str = "default") -> dict:
    cds_result = _build_ref_cds(source_id, cds_gff, ref_genome)
    if cds_result is None:
        return {}
    ref_cds, strand = cds_result
    ref_aa = _translate(ref_cds, strand)

    exons = cds_gff[cds_gff["gene_id"] == source_id].copy()
    exons = exons.sort_values("start", ascending=True)

    # Map queried variant positions to 0-based CDS offsets
    pos_info = {}
    for vi, pos in enumerate(meta["positions"]):
        cds_off = _genomic_to_cds_offset(int(pos), exons, strand)
        if cds_off is None:
            continue
        pos_info[str(pos)] = {"ref": str(meta["alleles"][vi][0]), "offset": cds_off}

    sorted_pos_info = sorted(pos_info.items(), key=lambda x: x[1]["offset"])

    cds_len = len(ref_cds)
    aa_to_pos: dict[int, list[str]] = {}
    for pos_str, info in pos_info.items():
        if strand == "+":
            aa_pos = info["offset"] // 3 + 1
        else:
            aa_pos = (cds_len - 1 - info["offset"]) // 3 + 1
        aa_to_pos.setdefault(aa_pos, []).append(pos_str)

    col_prefix = prefix if prefix else source_id

    # One column per individual AA position across all queried ranges
    all_aa_positions = sorted({
        aa_pos
        for aa_start, aa_end in aa_ranges
        for aa_pos in range(aa_start, aa_end + 1)
    })
    # Skip individual-position columns for very wide (e.g. full-gene) queries
    _gen_per_pos = len(all_aa_positions) <= _MAX_PER_POS_COLS
    per_pos_lists: dict[str, list] = (
        {f"{col_prefix}_{p}": [] for p in all_aa_positions} if _gen_per_pos else {}
    )

    haplotypes     = []
    ns_changes_col = []

    for _, row in deduped.iterrows():
        alleles_here = {p: row[p] for p in pos_info if p in deduped.columns}

        # ordered_ad positions hold "major{sep}minor" nucleotide pairs — pre-compute
        # separate major/minor translations so per-position columns can show "S/A".
        ordered_pos = {p for p, v in alleles_here.items()
                       if isinstance(v, str) and het_sep in v}
        if ordered_pos:
            alleles_major = {p: v.split(het_sep)[0] if p in ordered_pos else v
                             for p, v in alleles_here.items()}
            alleles_minor = {p: v.split(het_sep)[1] if p in ordered_pos else v
                             for p, v in alleles_here.items()}
            alt_aa_major = _translate(_apply_variants(ref_cds, sorted_pos_info, alleles_major), strand)
            alt_aa_minor = _translate(_apply_variants(ref_cds, sorted_pos_info, alleles_minor), strand)
        else:
            alt_aa_major = alt_aa_minor = None

        # het_sep alleles are treated as reference by _apply_variants
        alt_cds = _apply_variants(ref_cds, sorted_pos_info, alleles_here)
        alt_aa  = _translate(alt_cds, strand)

        hap_parts = []
        ns_list   = []

        for aa_start, aa_end in aa_ranges:
            range_pos     = [p for aa_p in range(aa_start, aa_end + 1)
                             for p in aa_to_pos.get(aa_p, [])]
            range_alleles = [alleles_here.get(p) for p in range_pos]

            has_missing  = "-" in range_alleles
            has_het      = HET_SYMBOL in range_alleles
            has_multi    = any(isinstance(v, str) and "," in v for v in range_alleles)
            has_ordered  = any(p in ordered_pos for p in range_pos)

            if has_missing:
                # Show per-position: "-" for missing positions, actual AA for present ones
                chars = []
                for aa_p in range(aa_start, aa_end + 1):
                    cvars = aa_to_pos.get(aa_p, [])
                    if any(alleles_here.get(p) == "-" for p in cvars):
                        chars.append("-")
                    else:
                        chars.append(_aa_at(alt_aa, aa_p))
                hap_parts.append("".join(chars))
            elif has_ordered and mode in ("default", "wide"):
                # Build char-by-char: het positions show [major/minor]
                chars = []
                for aa_p in range(aa_start, aa_end + 1):
                    cvars = aa_to_pos.get(aa_p, [])
                    if any(p in ordered_pos for p in cvars):
                        aa_m = _aa_at(alt_aa_major, aa_p)
                        aa_n = _aa_at(alt_aa_minor, aa_p)
                        chars.append(f"[{aa_m}/{aa_n}]" if aa_m != aa_n else aa_m)
                    else:
                        chars.append(_aa_at(alt_aa, aa_p))
                hap_parts.append("".join(chars))
            elif has_het or has_ordered or has_multi:
                # skip/collapse modes: collapse the whole range to "*"
                hap_parts.append(HET_SYMBOL)
            else:
                hap_parts.append(_aa_slice(alt_aa, aa_start, aa_end)
                                  if aa_start != aa_end else _aa_at(alt_aa, aa_start))

            # ns_changes for each individual AA position in this range
            for aa_pos in range(aa_start, aa_end + 1):
                codon_vars  = aa_to_pos.get(aa_pos, [])
                is_missing  = any(alleles_here.get(p) == "-" for p in codon_vars)
                is_het      = any(alleles_here.get(p) == HET_SYMBOL for p in codon_vars)
                is_ordered  = any(p in ordered_pos for p in codon_vars)
                ref_a = ref_aa[aa_pos - 1] if len(ref_aa) >= aa_pos else "?"
                alt_a = _aa_at(alt_aa, aa_pos)

                if is_missing:
                    ns_list.append(f"{ref_a}{aa_pos}-")
                elif is_het:
                    ns_list.append(f"{ref_a}{aa_pos}{HET_SYMBOL}")
                elif is_ordered:
                    aa_m = _aa_at(alt_aa_major, aa_pos)
                    aa_n = _aa_at(alt_aa_minor, aa_pos)
                    if aa_m == aa_n:
                        if aa_m != ref_a:
                            ns_list.append(f"{ref_a}{aa_pos}{aa_m}")
                    else:
                        ns_list.append(f"{ref_a}{aa_pos}{aa_m}{het_sep}{aa_n}")
                elif ref_a != alt_a:
                    ns_list.append(f"{ref_a}{aa_pos}{alt_a}")

        # Per-individual-position columns (skipped when _gen_per_pos is False)
        for aa_pos in (all_aa_positions if _gen_per_pos else []):
            col_name    = f"{col_prefix}_{aa_pos}"
            codon_vars  = aa_to_pos.get(aa_pos, [])
            pos_alleles = [alleles_here.get(p) for p in codon_vars]

            has_missing = "-" in pos_alleles
            has_het     = HET_SYMBOL in pos_alleles
            has_multi   = any(isinstance(v, str) and "," in v for v in pos_alleles)
            is_ordered  = any(p in ordered_pos for p in codon_vars)

            if has_missing:
                per_pos_lists[col_name].append("-")
            elif has_het:
                per_pos_lists[col_name].append(HET_SYMBOL)
            elif is_ordered:
                aa_m = _aa_at(alt_aa_major, aa_pos)
                aa_n = _aa_at(alt_aa_minor, aa_pos)
                per_pos_lists[col_name].append(f"{aa_m},{aa_n}" if aa_m != aa_n else aa_m)
            elif has_multi:
                multi_vars = {p: v for p in codon_vars
                              if isinstance((v := alleles_here.get(p, "")), str) and "," in v}
                n_opts = max(len(v.split(",")) for v in multi_vars.values())
                aa_opts = []
                for idx in range(n_opts):
                    test = dict(alleles_here)
                    for p, v in multi_vars.items():
                        opts = v.split(",")
                        test[p] = opts[idx] if idx < len(opts) else opts[-1]
                    alt_cds_t = _apply_variants(ref_cds, sorted_pos_info, test)
                    alt_aa_t  = _translate(alt_cds_t, strand)
                    aa_opts.append(_aa_at(alt_aa_t, aa_pos))
                per_pos_lists[col_name].append(",".join(aa_opts))
            else:
                per_pos_lists[col_name].append(_aa_at(alt_aa, aa_pos))

        haplotypes.append(",".join(hap_parts))
        ns_changes_col.append(ns_list)  # store as actual list, not str

    pos_sum = _pos_summary(aa_ranges)
    return {
        f"{col_prefix}_{pos_sum}_haplotype":  haplotypes,
        f"{col_prefix}_{pos_sum}_ns_changes": ns_changes_col,
        **per_pos_lists,
    }


# ── NT loci ───────────────────────────────────────────────────────────────────

def _add_nt_haplotypes(deduped, source_id, meta, nt_ranges,
                       intervals, ref_genome, het_sep: str = "/",
                       prefix: str | None = None) -> dict:
    # Build per-interval pos_info with offsets local to each interval
    interval_pos_infos = []

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
                    "offset": gpos - iv_start,
                }

        interval_pos_infos.append(
            sorted(local_pos_info.items(), key=lambda x: x[1]["offset"])
        )

    col_prefix = prefix if prefix else source_id

    # Per-interval column names (align with nt_ranges if available, else use intervals)
    range_source = nt_ranges if nt_ranges else [(iv[1], iv[2]) for iv in intervals]
    interval_col_names = [
        _range_col_name(col_prefix, s, e) for s, e in range_source
    ]
    # Pad/trim to match number of intervals
    while len(interval_col_names) < len(intervals):
        iv = intervals[len(interval_col_names)]
        interval_col_names.append(_range_col_name(col_prefix, iv[1], iv[2]))
    interval_col_names = interval_col_names[:len(intervals)]

    per_iv_lists: dict[str, list] = {cn: [] for cn in interval_col_names}

    haplotypes = []

    for _, row in deduped.iterrows():
        all_pos = {p for infos in interval_pos_infos for p, _ in infos}
        alleles_here = {p: row[p] for p in all_pos if p in deduped.columns}
        has_missing  = any(v == "-" for v in alleles_here.values())
        has_het      = any(v == HET_SYMBOL for v in alleles_here.values())

        if has_missing:
            haplotypes.append("-")
            for cn in interval_col_names:
                per_iv_lists[cn].append("-")
            continue

        # ordered_ad positions hold "major{sep}minor" nucleotide pairs
        ordered_pos = {p for p, v in alleles_here.items()
                       if isinstance(v, str) and het_sep in v}
        if ordered_pos:
            alleles_major = {p: v.split(het_sep)[0] if p in ordered_pos else v
                             for p, v in alleles_here.items()}
            alleles_minor = {p: v.split(het_sep)[1] if p in ordered_pos else v
                             for p, v in alleles_here.items()}

        parts = []
        for (chrom, iv_start, iv_end), sorted_pos_info, col_name in zip(
                intervals, interval_pos_infos, interval_col_names):
            ref_iv     = ref_genome[chrom][iv_start - 1 : iv_end]
            iv_pos_set = {p for p, _ in sorted_pos_info}
            iv_ordered = ordered_pos & iv_pos_set

            if iv_ordered:
                alt_major = _apply_variants(ref_iv, sorted_pos_info, alleles_major)
                alt_minor = _apply_variants(ref_iv, sorted_pos_info, alleles_minor)
                # haplotype keeps het_sep ("/") to stay unambiguous with the "," interval separator
                alt_seq_hap = f"{alt_major}{het_sep}{alt_minor}" if alt_major != alt_minor else alt_major
                alt_seq_col = f"{alt_major},{alt_minor}" if alt_major != alt_minor else alt_major
            else:
                alt_seq_hap = _apply_variants(ref_iv, sorted_pos_info, alleles_here)
                alt_seq_col = alt_seq_hap

            parts.append(alt_seq_hap)
            per_iv_lists[col_name].append(alt_seq_col)

        suffix = HET_SYMBOL if has_het else ""
        haplotypes.append(",".join(parts) + suffix)

    range_source_for_summary = nt_ranges if nt_ranges else [(iv[1], iv[2]) for iv in intervals]
    pos_sum = _pos_summary(range_source_for_summary)
    return {
        f"{col_prefix}_{pos_sum}_haplotype": haplotypes,
        **per_iv_lists,
    }
