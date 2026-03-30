import ast
import os
import re
import time
import datetime
import pandas as pd
import numpy as np
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, expand_full_gene_loci, resolve_loci, build_regions,
    build_variant_rows, load_call_data, build_allele_matrix,
)
from src.haplotypes import deduplicate_allele_matrix, compute_haplotypes

HAPLOTYPES_DIR = "haplotypes"


def _make_per_sample_df(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw
        .explode("sample_ids")
        .rename(columns={"sample_ids": "sample_id"})
        .drop(columns=["n_samples"])
        .reset_index(drop=True)
    )


def _make_haplotype_filename(parsed_loci: pd.DataFrame, format_mode: str) -> str:
    """Build a descriptive filename encoding loci, coord type, and format strategy.

    Pattern: {loci}__{coord_type}__{format_mode}__{YYYYMMDD_HHMMSS}.tsv
    Field separator __ avoids collision with underscores inside gene IDs.
    Loci with aliases use the alias; bare gene IDs are shortened to the
    portion after the PF3D7_ / Pf3D7_ prefix.
    Multiple loci are joined with +.
    """
    loci_labels = []
    for chrom, group in parsed_loci.groupby("chrom", sort=False):
        aliases = group["alias"].dropna().unique()
        if len(aliases) > 0:
            loci_labels.append(aliases[0])
        else:
            # PF3D7_0709000 → 0709000   |   Pf3D7_04_v3 → 04_v3
            m = re.match(r'^(?:PF3D7_|Pf3D7_)(.+)$', str(chrom), re.IGNORECASE)
            loci_labels.append(m.group(1) if m else str(chrom))

    loci_str = "+".join(loci_labels)
    coord_types = parsed_loci["coord_type"].unique()
    coord_str = coord_types[0] if len(coord_types) == 1 else "mixed"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{loci_str}__{coord_str}__{format_mode}__{ts}.tsv"


_NS_HET_RE = re.compile(r'^([A-Za-z?])(\d+)([A-Za-z*\-])\/([A-Za-z*\-])$')


def _format_ns_changes(val, mode: str) -> str:
    """Format an ns_changes cell (list or str repr) for display.

    Modes:
      default  – M56T/M  → M56[T/M],  joined with ", "
      skip     – M56*    → M56*,       joined with ", "
      collapse – M56T    → M56T,       joined with ", "
      wide     – M56T/M  → M56T, M56M, joined with ", "
    """
    if isinstance(val, list):
        changes = val
    else:
        try:
            changes = ast.literal_eval(str(val))
        except Exception:
            return str(val)
    if not isinstance(changes, list):
        return str(val)
    if not changes:
        return ""

    parts = []
    for entry in changes:
        m = _NS_HET_RE.match(entry)
        if m and mode == "default":
            parts.append(f"{m.group(1)}{m.group(2)}[{m.group(3)}/{m.group(4)}]")
        elif m and mode == "wide":
            parts.append(f"{m.group(1)}{m.group(2)}{m.group(3)}")
            parts.append(f"{m.group(1)}{m.group(2)}{m.group(4)}")
        else:
            parts.append(entry)
    return ", ".join(parts)


st.title("Order Haplotypes", anchor=False)

# ── Load static data ──────────────────────────────────────────────────────────
chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── User input ────────────────────────────────────────────────────────────────

DEBUG = st.toggle("Debug mode", value=False)

RAW_USER_INPUT = st.text_area(
    "Enter genomic loci",
    value="PF3D7_0709000[72-76,220,271] Pf3D7_04_v3[104205,139150-139156]",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · use `[*]` for the full gene · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Optional alias: `PF3D7_0709000[72,74](crt)` → columns named `crt_72`, `crt_74` · "
        "Multiple loci separated by spaces."
    ),
)

parsed_loci   = parse_loci_from_input(RAW_USER_INPUT)
parsed_loci   = expand_full_gene_loci(parsed_loci, reference_files["cds_gff"])
resolved_loci = resolve_loci(parsed_loci, reference_files["cds_gff"])

if parsed_loci.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()


if DEBUG:
    with st.expander("Debug"):
        st.write("**Chunk index** (all chromosomes in zarr)")
        st.dataframe(chunk_index_df, width="stretch", hide_index=True)

        st.write("**Parsed loci:**")
        st.dataframe(parsed_loci, width="stretch", hide_index=True)

        st.write("**Resolved genomic intervals:**")
        for sid, info in resolved_loci.items():
            st.text(f"{sid} ({info['coord_type']}): {info['intervals']}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Variant metadata table
# ══════════════════════════════════════════════════════════════════════════════

regions, warnings = build_regions(resolved_loci, variant_data, chunk_index_df)
for w in warnings:
    st.warning(w)

if not regions:
    st.warning("No variants found for any queried locus.")
    st.stop()

# ── Variant metadata table ────────────────────────────────────────────────────
st.divider()
st.subheader("Overview of variants of interest")

meta_rows = []
for source_id, region in regions.items():
    meta_rows.extend(build_variant_rows(source_id, region["meta"], resolved_loci[source_id]))

meta_df    = pd.DataFrame(meta_rows)
n_samples  = next(iter(regions.values()))["meta"]["n_samples"]
total_vars = sum(r["meta"]["n_variants"] for r in regions.values())

# ── Filter toggles ────────────────────────────────────────────────────────────
fcol1, fcol2, _ = st.columns(3)
apply_filter_pass = fcol1.toggle(
    "Filter pass variants only",
    value=False,
    help="Exclude variants that do not pass quality filters",
)
apply_numalt1 = fcol2.toggle(
    "Biallelic variants only",
    value=False,
    help="Exclude multi-allelic variant sites",
)

fail_fp  = ~meta_df["filter_pass"] if apply_filter_pass else pd.Series(False, index=meta_df.index)
fail_na  = (meta_df["numalt"] != 1) if apply_numalt1    else pd.Series(False, index=meta_df.index)
fail_any = fail_fp | fail_na

excluded_positions: set[str] = {
    str(meta_df.loc[i, "position"]) for i in meta_df.index if fail_any.loc[i]
}

# ── Styled display ────────────────────────────────────────────────────────────
_BOOL_COLS = ["is_snp", "filter_pass", "CDS", "needs_translation"]

def _highlight_failed(row):
    color = "background-color: #ffcccc" if fail_any.loc[row.name] else ""
    return [color] * len(row)

def _color_bool(val):
    if val == "✓":
        return "color: #2d8a4e"
    if val == "✗":
        return "color: #cc3333"
    return ""

display_meta = meta_df.copy()
for col in _BOOL_COLS:
    if col in display_meta.columns:
        display_meta[col] = display_meta[col].map({True: "✓", False: "✗"})

styled_meta = (
    display_meta.style
    .apply(_highlight_failed, axis=1)
    .map(_color_bool, subset=_BOOL_COLS)
)
st.dataframe(
    styled_meta,
    hide_index=True,
    width="stretch",
    column_config={
        "alt":    st.column_config.ListColumn("alt alleles"),
        "numalt": None,
    },
)
st.caption(
    f"{total_vars} variant sites across {len(regions)} {'locus' if len(regions) == 1 else 'loci'}, "
    f"affecting {n_samples:,} samples."
    + (" Rows highlighted in red are excluded from downstream steps." if fail_any.any() else ""),
    text_alignment="right",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Haplotypes
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Build haplotypes")

_FORMAT_DF = pd.DataFrame([
    {
        "format strategy": "default",
        "description": "Show both alleles at het positions with the major allele first",
        "speed": "slower", "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,[T/M]GK", "format of '_ns_changes' column": "G42-, K43T, M56[T/M], C58K"
    },{
        "format strategy": "skip",
        "description": "Skip processing hets and mark as *",
        "speed": "faster", "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "*",
        "format of '_haplotype' column": "-T,*GK", "format of '_ns_changes' column": "G42-, K43T, M56*, C58K"
    },{
        "format strategy": "collapse",
        "description": "Resolve het to hom using the major allele",
        "speed": "slower", "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,TGK", "format of '_ns_changes' column": "G42-, K43T, M56T, C58K"
    },{
        "format strategy": "wide",
        "description": "Expand het positions to two separate ns_changes entries",
        "speed": "slower", "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,[T/M]GK",  "format of '_ns_changes' column": "G42-, K43T, M56T, M56M, C58K"
    },
])

st.caption(
    "Imagine you queried `PF3D7_XXXXXXX[42-43, 56-58]` and a sample had: "
    "G42- (missing), K43T, M56[T/M], wild type G57, C58K — "
    "how would you like the output formatted?"
)
_fmt_selection = st.dataframe(
    _FORMAT_DF,
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)
_selected_rows = _fmt_selection.selection.rows
FORMAT_MODE = _FORMAT_DF.iloc[_selected_rows[0]]["format strategy"] if _selected_rows else "default"

_HET_MODE_MAP = {"default": "ordered_ad", "skip": "exclude",
                 "collapse": "major_ad",   "wide": "ordered_ad"}
HET_MODE = _HET_MODE_MAP[FORMAT_MODE]
HET_SEP  = "/"

# Small state management section
_load_state = (RAW_USER_INPUT, apply_filter_pass, apply_numalt1, FORMAT_MODE)
if st.session_state.get("last_load_state") != _load_state:
    st.session_state["last_load_state"]    = _load_state
    st.session_state["haplotypes_built"]   = False
    st.session_state.pop("haplotypes_raw", None)
    st.session_state.pop("haplotypes_saved_path", None)

if not st.session_state.get("haplotypes_built"):
    if _selected_rows and st.button("Build haplotypes", type="primary"):
        st.session_state["haplotypes_built"] = True
        st.rerun()
else:
    # ── Compute (once per unique input state) ─────────────────────────────────
    if "haplotypes_raw" not in st.session_state:
        _save_intermediates = st.session_state.get("_debug_save_intermediates", False)

        t0 = time.time()
        partial_matrices = []
        region_ids = list(regions.keys())
        n_regions  = len(region_ids)
        n_steps    = n_regions + 2  # +dedup +compute
        progress   = st.progress(0, text="Loading genotypes…")

        for i, source_id in enumerate(region_ids):
            region = regions[source_id]
            progress.progress(i / n_steps, text=f"Loading {source_id} ({i + 1}/{n_regions})…")
            region["genotypes"], region["g1_wins"] = load_call_data(
                region["ds"],
                cache_key=(source_id, apply_filter_pass, apply_numalt1),
                load_ad=(HET_MODE in ("major_ad", "ordered_ad")),
            )
            partial_matrices.append(
                build_allele_matrix({source_id: region}, excluded_positions, HET_MODE, HET_SEP)
            )
            region["genotypes"] = None
            region["g1_wins"]   = None

        progress.progress(n_regions / n_steps, text="Deduplicating allele matrix…")
        allele_matrix = pd.concat(partial_matrices, axis=1)
        partial_matrices.clear()

        if _save_intermediates:
            st.session_state["_debug_allele_matrix"] = allele_matrix.copy()

        deduped = deduplicate_allele_matrix(allele_matrix)
        allele_matrix = None

        if _save_intermediates:
            st.session_state["_debug_deduped"] = deduped

        progress.progress((n_regions + 1) / n_steps, text="Computing haplotypes…")
        raw = compute_haplotypes(
            deduped, regions, resolved_loci, parsed_loci,
            reference_files["cds_gff"], het_sep=HET_SEP, mode=FORMAT_MODE,
        )

        # Drop raw position columns before caching
        _pos_cols = [c for c in raw.columns if str(c).isdigit()]
        raw = raw.drop(columns=_pos_cols)
        deduped = None

        # ── Format ns_changes before saving ───────────────────────────────────
        ns_cols = [c for c in raw.columns if c.endswith("_ns_changes")]
        raw_for_save = raw.copy()
        for _c in ns_cols:
            raw_for_save[_c] = raw_for_save[_c].apply(
                lambda v: _format_ns_changes(v, FORMAT_MODE)
            )

        # ── Save to haplotypes directory ───────────────────────────────────────
        os.makedirs(HAPLOTYPES_DIR, exist_ok=True)
        fname = _make_haplotype_filename(parsed_loci, FORMAT_MODE)
        fpath = os.path.join(HAPLOTYPES_DIR, fname)
        _make_per_sample_df(raw_for_save).to_csv(fpath, sep="\t", index=False)

        progress.progress(1.0, text="Done!")
        progress.empty()

        st.session_state["haplotypes_raw"]        = raw
        st.session_state["_hap_elapsed"]          = time.time() - t0
        st.session_state["haplotypes_saved_path"] = fpath

    raw     = st.session_state["haplotypes_raw"]
    deduped = st.session_state.get("_debug_deduped")
    elapsed = st.session_state["_hap_elapsed"]
    saved_path = st.session_state.get("haplotypes_saved_path")

    if saved_path:
        st.success(
            f"Built in {elapsed:.1f}s — saved to `{saved_path}`. "
            "Head to the **Inspect** page to browse and combine files."
        )
    else:
        st.success(f"Loaded in {elapsed:.1f}s")

    # ── Build display copy with ns_changes formatted per the earlier selection ──
    ns_cols     = [c for c in raw.columns if c.endswith("_ns_changes")]
    display_raw = raw.copy()
    for _c in ns_cols:
        display_raw[_c] = display_raw[_c].apply(
            lambda v: _format_ns_changes(v, FORMAT_MODE)
        )

    # ── Download button ────────────────────────────────────────────────────────
    _tsv = _make_per_sample_df(display_raw).to_csv(sep="\t", index=False)
    st.download_button(
        "Download TSV",
        _tsv,
        file_name=os.path.basename(saved_path) if saved_path else "haplotypes.tsv",
        mime="text/tab-separated-values",
    )

    if DEBUG:
        with st.expander("Debug"):
            st.toggle(
                "Save intermediate outputs",
                key="_debug_save_intermediates",
                help="When on, the allele matrix is preserved in session state before being freed. "
                     "Flip on, then re-run by changing any input.",
            )
            if st.session_state.get("_debug_save_intermediates") and "_debug_allele_matrix" in st.session_state:
                st.write("**Allele matrix** (samples × positions)")
                st.dataframe(st.session_state["_debug_allele_matrix"], hide_index=False, width="stretch")

            if deduped is not None:
                st.write("**Deduplicated allele matrix**")
                st.dataframe(deduped, hide_index=True, width="stretch")

            st.write("**Haplotype output**")
            st.dataframe(display_raw, hide_index=True, width="stretch")
