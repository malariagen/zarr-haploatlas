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


def _parse_filename(fname: str) -> dict:
    m = _FNAME_RE.match(fname)
    if m:
        try:
            dt = datetime.datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
            created = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            created = m.group("ts")
        return {
            "filename":   fname,
            "loci":       m.group("loci").replace("+", " + "),
            "type":       m.group("coord_type"),
            "format":     m.group("fmt"),
            "created":    created,
        }
    # Fallback for files that don't match (e.g. manually placed reference files)
    return {
        "filename": fname,
        "loci":     fname.removesuffix(".tsv"),
        "type":     "?",
        "format":   "?",
        "created":  "?",
    }


def _safe_col_prefix(loci_str: str) -> str:
    """Turn a loci label like 'crt + dhps' into a safe column prefix 'crt+dhps'."""
    return re.sub(r'[^A-Za-z0-9+]', '_', loci_str.replace(" + ", "+"))


# ── Page ─────────────────────────────────────────────────────────────────────

st.title("Inspect Haplotypes", anchor=False)
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
    use_container_width=True,
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "filename": st.column_config.TextColumn("Filename"),
        "loci":     st.column_config.TextColumn("Loci",    width="medium"),
        "type":     st.column_config.TextColumn("Type",    width="small"),
        "format":   st.column_config.TextColumn("Format",  width="small"),
        "created":  st.column_config.TextColumn("Created", width="medium"),
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
    loci_labels = [files_df.iloc[i]["loci"] for i in selected_indices]
    st.subheader(f"Selected {n_sel} files: {', '.join(loci_labels)}")

if st.button("Merge & Download", type="primary"):
    # ── Load files ────────────────────────────────────────────────────────────
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
    if not dfs:
        st.stop()

    # ── Check for column conflicts across files ───────────────────────────────
    # A conflict exists when the same non-sample_id column appears in 2+ files.
    all_non_sid = [set(df.columns) - {"sample_id"} for df in dfs]
    all_non_sid_union = set().union(*all_non_sid)
    conflict_cols = {
        c for c in all_non_sid_union
        if sum(c in cols for cols in all_non_sid) > 1
    }

    if conflict_cols:
        # Prefix each file's non-sample_id columns with its loci label to
        # disambiguate (e.g. two CRT files with different format modes).
        labeled_dfs = []
        for df, row_idx in zip(dfs, selected_indices):
            prefix = _safe_col_prefix(files_df.iloc[row_idx]["loci"])
            rename = {
                c: f"{c}__{prefix}"
                for c in df.columns
                if c != "sample_id" and c in conflict_cols
            }
            labeled_dfs.append(df.rename(columns=rename))
        dfs = labeled_dfs

    # ── Merge all on sample_id (outer join) ───────────────────────────────────
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="sample_id", how="outer")

    # Put sample_id first, sort rows alphabetically
    cols = ["sample_id"] + [c for c in merged.columns if c != "sample_id"]
    merged = merged[cols].sort_values("sample_id").reset_index(drop=True)

    # ── Build download filename ───────────────────────────────────────────────
    all_loci = "+".join(
        _safe_col_prefix(files_df.iloc[i]["loci"]) for i in selected_indices
    )
    ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if n_sel == 1:
        download_name = selected_files[0]
    else:
        download_name = f"merged__{all_loci}__{ts_now}.tsv"

    # ── Output ────────────────────────────────────────────────────────────────
    st.success(
        f"Merged {n_sel} file(s): "
        f"{merged.shape[0]:,} samples × {merged.shape[1]} columns."
    )

    tsv_out = merged.to_csv(sep="\t", index=False)
    st.download_button(
        "Download merged TSV",
        tsv_out,
        file_name=download_name,
        mime="text/tab-separated-values",
    )

    with st.expander("Preview (first 20 rows)"):
        st.dataframe(merged.head(20), hide_index=True, use_container_width=True)
