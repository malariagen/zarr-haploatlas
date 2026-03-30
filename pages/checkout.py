import os
import re
import datetime
import pandas as pd
import streamlit as st

HAPLOTYPES_DIR = "haplotypes"

# ── Filename parsing ──────────────────────────────────────────────────────────
# Expected pattern: {loci}__{coord_type}__{format_mode}__{YYYYMMDD_HHMMSS}.tsv
# Double-underscore is the field separator; gene names may contain single underscores.

_FNAME_RE = re.compile(
    r'^(?P<loci>.+?)__(?P<coord_type>aa|nt|mixed)__(?P<fmt>default|skip|collapse|wide)__'
    r'(?P<ts>\d{8}_\d{6})\.tsv$'
)
# loci field has the form {gene_label}_{pos_summary}, e.g. crt_72.74 or 0709000_76-93.
# We split on the *last* underscore to separate gene label from position summary.
_LOCI_SPLIT_RE = re.compile(r'^(.+)_([^_]+)$')


def _parse_filename(fname: str) -> dict:
    m = _FNAME_RE.match(fname)
    if m:
        try:
            dt = datetime.datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
            created = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            created = m.group("ts")
        loci_raw = m.group("loci")   # e.g. "crt_72.74" or "0709000_76-93"
        loci_m = _LOCI_SPLIT_RE.match(loci_raw)
        gene     = loci_m.group(1) if loci_m else loci_raw
        pos      = loci_m.group(2).replace(".", ", ") if loci_m else ""
        return {
            "filename": fname,
            "gene":     gene,
            "positions": pos,
            "type":     m.group("coord_type"),
            "format":   m.group("fmt"),
            "created":  created,
            # keep raw loci for merge logic
            "_loci_raw": loci_raw,
        }
    # Fallback for manually placed files that don't match the naming convention
    return {
        "filename":  fname,
        "gene":      fname.removesuffix(".tsv"),
        "positions": "",
        "type":      "?",
        "format":    "?",
        "created":   "?",
        "_loci_raw": fname.removesuffix(".tsv"),
    }


def _safe_col_prefix(loci_raw: str) -> str:
    """Turn a raw loci string like 'crt_72.74' into a safe column prefix."""
    return re.sub(r'[^A-Za-z0-9]', '_', loci_raw)


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("Checkout", anchor=False)
st.caption("Browse saved haplotype files, select any combination, and download a single merged TSV.")

os.makedirs(HAPLOTYPES_DIR, exist_ok=True)

tsv_files = sorted(
    [f for f in os.listdir(HAPLOTYPES_DIR) if f.endswith(".tsv")],
    reverse=True,  # newest first
)

if not tsv_files:
    st.info(
        f"No haplotype files found in `{HAPLOTYPES_DIR}/`. "
        "Build some on the **Order** page first."
    )
    st.stop()

# ── File browser ──────────────────────────────────────────────────────────────
rows = [_parse_filename(f) for f in tsv_files]
files_df = pd.DataFrame(rows)

st.subheader(f"Available files ({len(tsv_files)})")
selection = st.dataframe(
    files_df,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "filename":  st.column_config.TextColumn("Filename"),
        "gene":      st.column_config.TextColumn("Gene / alias", width="medium"),
        "positions": st.column_config.TextColumn("Positions",    width="medium"),
        "type":      st.column_config.TextColumn("Type",         width="small"),
        "format":    st.column_config.TextColumn("Format",       width="small"),
        "created":   st.column_config.TextColumn("Created",      width="medium"),
        "_loci_raw": None,  # hide internal field
    },
)

selected_indices = selection.selection.rows
selected_files   = [files_df.iloc[i]["filename"] for i in selected_indices]

if not selected_files:
    st.caption("Select one or more files above, then merge and download.")
    st.stop()

# ── Merge controls ────────────────────────────────────────────────────────────
st.divider()

n_sel = len(selected_files)
if n_sel == 1:
    st.subheader(f"Selected: `{selected_files[0]}`")
else:
    gene_labels = [files_df.iloc[i]["gene"] for i in selected_indices]
    st.subheader(f"Selected {n_sel} files: {', '.join(gene_labels)}")

# ── Merge ─────────────────────────────────────────────────────────────────────
# Cache key: the sorted set of selected filenames. If the selection changes,
# the cached merge is stale and we show the Merge button again.
_merge_cache_key = tuple(sorted(selected_files))
_merge_ready = (
    st.session_state.get("checkout_merge_key") == _merge_cache_key
    and "checkout_merged_df" in st.session_state
)

if not _merge_ready:
    if st.button("Merge", type="primary"):
        dfs = []
        load_errors = []
        for fname in selected_files:
            path = os.path.join(HAPLOTYPES_DIR, fname)
            try:
                dfs.append(pd.read_csv(path, sep="\t", low_memory=False))
            except Exception as e:
                load_errors.append(f"`{fname}`: {e}")

        for err in load_errors:
            st.error(err)

        if dfs:
            # Detect column conflicts (same non-sample_id column in 2+ files)
            all_non_sid = [set(df.columns) - {"sample_id"} for df in dfs]
            conflict_cols = {
                c for c in set().union(*all_non_sid)
                if sum(c in cols for cols in all_non_sid) > 1
            }
            if conflict_cols:
                labeled_dfs = []
                for df, row_idx in zip(dfs, selected_indices):
                    prefix = _safe_col_prefix(files_df.iloc[row_idx]["_loci_raw"])
                    rename = {
                        c: f"{c}__{prefix}"
                        for c in df.columns
                        if c != "sample_id" and c in conflict_cols
                    }
                    labeled_dfs.append(df.rename(columns=rename))
                dfs = labeled_dfs

            merged = dfs[0]
            for df in dfs[1:]:
                merged = merged.merge(df, on="sample_id", how="outer")

            cols = ["sample_id"] + [c for c in merged.columns if c != "sample_id"]
            merged = merged[cols].sort_values("sample_id").reset_index(drop=True)

            all_loci = "+".join(
                _safe_col_prefix(files_df.iloc[i]["_loci_raw"]) for i in selected_indices
            )
            ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            download_name = selected_files[0] if n_sel == 1 else f"merged__{all_loci}__{ts_now}.tsv"

            st.session_state["checkout_merged_df"]   = merged
            st.session_state["checkout_merged_name"] = download_name
            st.session_state["checkout_merge_key"]   = _merge_cache_key
            st.rerun()

# ── Preview + download (shown once merge result is cached) ────────────────────
if _merge_ready:
    merged        = st.session_state["checkout_merged_df"]
    download_name = st.session_state["checkout_merged_name"]

    st.caption(
        f"{merged.shape[0]:,} samples × {merged.shape[1]} columns"
        + (f" merged from {n_sel} files" if n_sel > 1 else "")
    )
    st.dataframe(merged, hide_index=True, width="stretch")

    tsv_out = merged.to_csv(sep="\t", index=False)
    st.download_button(
        "Download TSV",
        tsv_out,
        file_name=download_name,
        mime="text/tab-separated-values",
    )
