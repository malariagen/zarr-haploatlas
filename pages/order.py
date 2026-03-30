import ast
import hashlib
import os
import re
import time
import datetime
import pandas as pd
import streamlit as st

from src.utils import (
    build_chunk_index, load_reference_files, load_variant_data,
    parse_loci_from_input, expand_full_gene_loci, resolve_loci, build_regions,
    build_variant_rows, load_call_data, build_allele_matrix,
)
from src.haplotypes import deduplicate_allele_matrix, compute_haplotypes

HAPLOTYPES_DIR = "haplotypes"

# Regex to extract individual token strings from user input.
# Each token is one bracket expression: IDENTIFIER[positions](optional_alias)
_TOKEN_RE = re.compile(r'([^\s\[]+\[[^\]]+\](?:\([^)]+\))?)')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_per_sample_df(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw
        .explode("sample_ids")
        .rename(columns={"sample_ids": "sample_id"})
        .drop(columns=["n_samples"])
        .reset_index(drop=True)
    )


def _token_to_filename(token: str, format_mode: str) -> str:
    """Derive a filename from a single token string.

    Pattern: {gene_label}_{pos_summary}__{coord_type}__{format_mode}__{YYYYMMDD_HHMMSS}.tsv

    gene_label  = alias if given, else the portion of the gene/chrom ID after PF3D7_/Pf3D7_.
    pos_summary = positions with spaces stripped, commas replaced by dots, '*' → 'all'.
    Double-underscore (__) is the field separator; single underscores may appear within fields.

    Examples:
      PF3D7_0709000[72,74](crt)   → crt_72.74__aa__default__20260330_143022.tsv
      PF3D7_0709000[76-93]        → 0709000_76-93__aa__default__20260330_143023.tsv
      PF3D7_0709000[*](dhfr)      → dhfr_all__aa__default__20260330_143024.tsv
      Pf3D7_04_v3[104205,139150]  → 04_v3_104205.139150__nt__default__20260330_143025.tsv
    """
    alias_m = re.search(r'\(([^)]+)\)$', token)
    alias = alias_m.group(1) if alias_m else None
    body = token[:alias_m.start()].strip() if alias_m else token

    ident, pos_part = body.split("[", 1)
    pos_part = pos_part.rstrip("]").strip()

    is_gene = bool(re.match(r"^PF3D7_(?:\d{7}|(?:MIT|API)\d{5})$", ident, re.IGNORECASE))
    coord_type = "aa" if is_gene else "nt"

    if alias:
        gene_label = alias
    else:
        m = re.match(r'^(?:PF3D7_|Pf3D7_)(.+)$', ident, re.IGNORECASE)
        gene_label = m.group(1) if m else ident

    pos_summary = "all" if pos_part == "*" else re.sub(r'\s+', '', pos_part).replace(',', '.')

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{gene_label}_{pos_summary}__{coord_type}__{format_mode}__{ts}.tsv"


_NS_HET_RE = re.compile(r'^([A-Za-z?])(\d+)([A-Za-z*\-])\/([A-Za-z*\-])$')


def _format_ns_changes(val, mode: str) -> str:
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


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("Order Haplotypes", anchor=False)

chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()
variant_data    = load_variant_data()

# ── Initialise widget defaults once — never rely on value= to re-set these ───
# Streamlit can re-apply value= on every render in some versions, which would
# overwrite the user's in-session changes and corrupt the settings hash.
_WIDGET_DEFAULTS: dict = {
    "order_debug":        False,
    "order_loci_input":   "PF3D7_0709000[72-76,220,271] Pf3D7_04_v3[104205,139150-139156]",
    "order_filter_pass":  False,
    "order_numalt":       False,
}
for _k, _v in _WIDGET_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

DEBUG = st.toggle("Debug mode", key="order_debug")

RAW_USER_INPUT = st.text_area(
    "Enter genomic loci",
    key="order_loci_input",
    help=(
        "Amino acid: `PF3D7_XXXXXXX[start-end,pos]` · use `[*]` for the full gene · "
        "Nucleotide: `Pf3D7_??_v3[start-end,pos]` · "
        "Optional alias: `PF3D7_0709000[72,74](crt)` → columns named `crt_72`, `crt_74` · "
        "Multiple loci separated by spaces. Each bracket expression becomes one file."
    ),
)

# Parse all loci together for the variant overview table.
# Per-token parsing happens inside the build loop.
parsed_loci   = parse_loci_from_input(RAW_USER_INPUT)
parsed_loci   = expand_full_gene_loci(parsed_loci, reference_files["cds_gff"])
resolved_loci = resolve_loci(parsed_loci, reference_files["cds_gff"])

# Individual token strings — one file will be saved per token.
_tokens = _TOKEN_RE.findall(RAW_USER_INPUT)

if parsed_loci.empty:
    st.info("Enter one or more loci above to begin.")
    st.stop()

if DEBUG:
    with st.expander("Debug"):
        st.write("**Chunk index**")
        st.dataframe(chunk_index_df, width="stretch", hide_index=True)
        st.write("**Parsed loci (all tokens combined):**")
        st.dataframe(parsed_loci, width="stretch", hide_index=True)
        st.write("**Resolved genomic intervals:**")
        for sid, info in resolved_loci.items():
            st.text(f"{sid} ({info['coord_type']}): {info['intervals']}")
        st.write(f"**Tokens ({len(_tokens)}):** {_tokens}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. Variant metadata table (all tokens combined for overview)
# ══════════════════════════════════════════════════════════════════════════════

regions, warnings = build_regions(resolved_loci, variant_data, chunk_index_df)
for w in warnings:
    st.warning(w)

if not regions:
    st.warning("No variants found for any queried locus.")
    st.stop()

st.divider()
st.subheader("Overview of variants of interest")

meta_rows = []
for source_id, region in regions.items():
    meta_rows.extend(build_variant_rows(source_id, region["meta"], resolved_loci[source_id]))

meta_df    = pd.DataFrame(meta_rows)
n_samples  = next(iter(regions.values()))["meta"]["n_samples"]
total_vars = sum(r["meta"]["n_variants"] for r in regions.values())

fcol1, fcol2, _ = st.columns(3)
apply_filter_pass = fcol1.toggle(
    "Filter pass variants only",
    key="order_filter_pass",
    help="Exclude variants that do not pass quality filters",
)
apply_numalt1 = fcol2.toggle(
    "Biallelic variants only",
    key="order_numalt",
    help="Exclude multi-allelic variant sites",
)

fail_fp  = ~meta_df["filter_pass"] if apply_filter_pass else pd.Series(False, index=meta_df.index)
fail_na  = (meta_df["numalt"] != 1) if apply_numalt1    else pd.Series(False, index=meta_df.index)
fail_any = fail_fp | fail_na

excluded_positions: set[str] = {
    str(meta_df.loc[i, "position"]) for i in meta_df.index if fail_any.loc[i]
}

_BOOL_COLS = ["is_snp", "filter_pass", "CDS", "needs_translation"]

def _highlight_failed(row):
    color = "background-color: #ffcccc" if fail_any.loc[row.name] else ""
    return [color] * len(row)

def _color_bool(val):
    if val == "✓": return "color: #2d8a4e"
    if val == "✗": return "color: #cc3333"
    return ""

display_meta = meta_df.copy()
for col in _BOOL_COLS:
    if col in display_meta.columns:
        display_meta[col] = display_meta[col].map({True: "✓", False: "✗"})

st.dataframe(
    display_meta.style.apply(_highlight_failed, axis=1).map(_color_bool, subset=_BOOL_COLS),
    hide_index=True,
    width="stretch",
    column_config={"alt": st.column_config.ListColumn("alt alleles"), "numalt": None},
)
st.caption(
    f"{total_vars} variant sites across {len(regions)} {'locus' if len(regions) == 1 else 'loci'}, "
    f"affecting {n_samples:,} samples."
    + (" Rows highlighted in red are excluded from downstream steps." if fail_any.any() else ""),
    text_alignment="right",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Haplotype build
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("Build haplotypes")

_FORMAT_DF = pd.DataFrame([
    {
        "format strategy": "default",
        "description": "Show both alleles at het positions with the major allele first",
        "speed": "slower",
        "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,[T/M]GK",
        "format of '_ns_changes' column": "G42-, K43T, M56[T/M], C58K",
    }, {
        "format strategy": "skip",
        "description": "Skip processing hets and mark as *",
        "speed": "faster",
        "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "*",
        "format of '_haplotype' column": "-T,*GK",
        "format of '_ns_changes' column": "G42-, K43T, M56*, C58K",
    }, {
        "format strategy": "collapse",
        "description": "Resolve het to hom using the major allele",
        "speed": "slower",
        "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,TGK",
        "format of '_ns_changes' column": "G42-, K43T, M56T, C58K",
    }, {
        "format strategy": "wide",
        "description": "Expand het positions to two separate ns_changes entries",
        "speed": "slower",
        "format of mutation position column, e.g., PF3D7_XXXXXXX_56": "T,M",
        "format of '_haplotype' column": "-T,[T/M]GK",
        "format of '_ns_changes' column": "G42-, K43T, M56T, M56M, C58K",
    },
])

st.caption(
    "Imagine you queried `PF3D7_XXXXXXX[42-43, 56-58]` and a sample had: "
    "G42- (missing), K43T, M56[T/M], wild type G57, C58K — "
    "how would you like the output formatted?"
)
_fmt_sel = st.dataframe(
    _FORMAT_DF,
    key="order_fmt_selection",
    hide_index=True,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)
_selected_rows = _fmt_sel.selection.rows
if _selected_rows:
    # User made (or restored) a selection — update the persisted value.
    FORMAT_MODE = _FORMAT_DF.iloc[_selected_rows[0]]["format strategy"]
    st.session_state["order_fmt_mode"] = FORMAT_MODE
else:
    # No selection visible (e.g. first render or selection lost on navigation) —
    # fall back to whatever was last explicitly chosen, defaulting to "default".
    FORMAT_MODE = st.session_state.get("order_fmt_mode", "default")

_HET_MODE_MAP = {"default": "ordered_ad", "skip": "exclude",
                 "collapse": "major_ad",   "wide": "ordered_ad"}
HET_MODE = _HET_MODE_MAP[FORMAT_MODE]
HET_SEP  = "/"

# ── Settings hash — invalidates per-token cache when any input changes ────────
_settings_hash = hashlib.md5(
    repr((RAW_USER_INPUT, apply_filter_pass, apply_numalt1, FORMAT_MODE)).encode()
).hexdigest()[:16]

if st.session_state.get("order_settings_hash") != _settings_hash:
    for k in [k for k in st.session_state if k.startswith("order_token_")]:
        del st.session_state[k]
    st.session_state["order_settings_hash"] = _settings_hash
    st.session_state["order_haplotypes_started"] = False

# ── Build button ──────────────────────────────────────────────────────────────
if not st.session_state.get("order_haplotypes_started"):
    # Allow building if there's a live selection OR a previously persisted mode
    # (the selection can be lost on navigation but FORMAT_MODE is still valid).
    _fmt_chosen = bool(_selected_rows) or "order_fmt_mode" in st.session_state
    if _fmt_chosen:
        n_tok = len(_tokens)
        label = f"Build haplotypes ({n_tok} file{'s' if n_tok != 1 else ''})"
        if st.button(label, type="primary"):
            st.session_state["order_haplotypes_started"] = True
            st.rerun()
    else:
        st.caption("Select a format strategy above to enable building.")

# ── Per-token build loop ──────────────────────────────────────────────────────
else:
    # Show status for all tokens already completed this session
    n_done = sum(1 for i in range(len(_tokens)) if f"order_token_{i}_path" in st.session_state)

    for i in range(n_done):
        path = st.session_state.get(f"order_token_{i}_path")
        token_label = _tokens[i] if i < len(_tokens) else f"token {i+1}"
        if path:
            st.success(f"({i+1}/{len(_tokens)}) `{token_label}` → saved to `{path}`")
        else:
            st.warning(f"({i+1}/{len(_tokens)}) `{token_label}` → no variants found, skipped.")

    # Find next pending token
    next_i = next(
        (i for i in range(len(_tokens)) if f"order_token_{i}_path" not in st.session_state),
        None,
    )

    if next_i is not None:
        token = _tokens[next_i]
        st.info(f"Building ({next_i + 1}/{len(_tokens)}): `{token}`…")

        # ── Parse this token independently ────────────────────────────────────
        parsed_t  = parse_loci_from_input(token)
        parsed_t  = expand_full_gene_loci(parsed_t, reference_files["cds_gff"])
        resolved_t = resolve_loci(parsed_t, reference_files["cds_gff"])

        if parsed_t.empty or not resolved_t:
            st.warning(f"Could not resolve `{token}` — skipping.")
            st.session_state[f"order_token_{next_i}_path"] = None
            st.rerun()
        else:
            regions_t, warns_t = build_regions(resolved_t, variant_data, chunk_index_df)
            for w in warns_t:
                st.warning(w)

            if not regions_t:
                st.warning(f"No variants found for `{token}` — skipping.")
                st.session_state[f"order_token_{next_i}_path"] = None
                st.rerun()
            else:
                t0       = time.time()
                rids     = list(regions_t.keys())
                n_r      = len(rids)
                n_steps  = n_r + 2
                progress = st.progress(0, text="Loading genotypes…")

                for j, sid in enumerate(rids):
                    region = regions_t[sid]
                    progress.progress(j / n_steps, text=f"Loading {sid}…")
                    # Include positions in the cache key: different tokens can query
                    # the same chromosome at different position ranges, producing
                    # different ds objects. Without positions, load_call_data would
                    # return a stale cache hit from an earlier token and cause an
                    # IndexError when the variant counts differ.
                    _pos_key = tuple(region["meta"]["positions"].tolist())
                    region["genotypes"], region["g1_wins"] = load_call_data(
                        region["ds"],
                        cache_key=(sid, apply_filter_pass, apply_numalt1, _pos_key),
                        load_ad=(HET_MODE in ("major_ad", "ordered_ad")),
                    )

                progress.progress(n_r / n_steps, text="Building allele matrix…")
                am = build_allele_matrix(regions_t, excluded_positions, HET_MODE, HET_SEP)
                for r in regions_t.values():
                    r["genotypes"] = r["g1_wins"] = None

                progress.progress((n_r + 1) / n_steps, text="Computing haplotypes…")
                deduped_t = deduplicate_allele_matrix(am)
                am = None

                raw_t = compute_haplotypes(
                    deduped_t, regions_t, resolved_t, parsed_t,
                    reference_files["cds_gff"], het_sep=HET_SEP, mode=FORMAT_MODE,
                )
                deduped_t = None

                # Drop raw position columns
                raw_t = raw_t.drop(columns=[c for c in raw_t.columns if str(c).isdigit()])

                # Format ns_changes
                for c in [c for c in raw_t.columns if c.endswith("_ns_changes")]:
                    raw_t[c] = raw_t[c].apply(lambda v: _format_ns_changes(v, FORMAT_MODE))

                # Save
                os.makedirs(HAPLOTYPES_DIR, exist_ok=True)
                fname = _token_to_filename(token, FORMAT_MODE)
                fpath = os.path.join(HAPLOTYPES_DIR, fname)
                _make_per_sample_df(raw_t).to_csv(fpath, sep="\t", index=False)

                elapsed = time.time() - t0
                progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
                progress.empty()

                st.session_state[f"order_token_{next_i}_path"] = fpath
                st.rerun()

    else:
        # All tokens finished
        st.caption(
            f"All {len(_tokens)} file{'s' if len(_tokens) != 1 else ''} saved. "
            "Switch to **Inspect** to browse and combine them."
        )

        if DEBUG:
            with st.expander("Debug"):
                st.toggle(
                    "Save intermediate outputs",
                    key="order_debug_intermediates",
                    help="Flip on, then rebuild to capture intermediate matrices.",
                )
                for i in range(len(_tokens)):
                    path = st.session_state.get(f"order_token_{i}_path")
                    if path:
                        st.write(f"Token {i}: `{path}`")
