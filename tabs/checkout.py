import os
import re
import datetime
import pandas as pd
import streamlit as st

from src.haplotype_summary import render_checkout_haplotype_summary

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


def _is_mutation_column(col_name: str) -> bool:
    if col_name == "sample_id":
        return False
    if col_name.endswith("_haplotype") or col_name.endswith("_ns_changes"):
        return False
    if col_name.startswith("Unnamed"):
        return False
    if re.search(r"_(\d+)(?:_\d+)?(?:__.*)?$", col_name):
        return True
    return False


def _candidate_geography_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c == "sample_id":
            continue
        if not pd.api.types.is_object_dtype(df[c]) and not pd.api.types.is_string_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) > 100:
            continue
        c_lower = c.lower()
        if any(k in c_lower for k in ["pop", "country", "site", "region", "geo"]):
            cols.append(c)
    return cols


# ── Page ─────────────────────────────────────────────────────────────────────

def render():
    st.title("Checkout", anchor=False)
    st.caption("Browse saved haplotype files, select any combination, and download a single merged TSV.")

    if st.button("Refresh file list"):
        st.rerun()

    os.makedirs(HAPLOTYPES_DIR, exist_ok=True)

    tsv_files = sorted(
        [f for f in os.listdir(HAPLOTYPES_DIR) if f.endswith(".tsv")],
        reverse=True,  # newest first
    )

    if not tsv_files:
        st.info(
            f"No haplotype files found in `{HAPLOTYPES_DIR}/`. "
            "Build some on the **Order** tab first."
        )
        return

    # ── File browser ──────────────────────────────────────────────────────────
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
        return

    # ── Merge controls ────────────────────────────────────────────────────────
    st.divider()

    n_sel = len(selected_files)
    if n_sel == 1:
        st.subheader(f"Selected: `{selected_files[0]}`")
    else:
        gene_labels = [files_df.iloc[i]["gene"] for i in selected_indices]
        st.subheader(f"Selected {n_sel} files: {', '.join(gene_labels)}")

    # ── Merge ─────────────────────────────────────────────────────────────────
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

    # ── Preview + download (shown once merge result is cached) ────────────────
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

        st.divider()
        st.subheader("Haplotype summary")
        st.caption(
            "Select mutation columns (for example `dhps_456`, `dhps_459`, `dhfr_724`) "
            "to build haplotype combinations and visualise them as a Bokeh haplotype summary view."
        )

        mutation_candidates = [c for c in merged.columns if _is_mutation_column(c)]
        if len(mutation_candidates) < 2:
            st.info("Need at least two mutation-like columns in the merged data to plot a haplotype summary.")
            return

        default_cols = mutation_candidates[: min(3, len(mutation_candidates))]
        selected_mutation_cols = st.multiselect(
            "Mutation columns",
            options=mutation_candidates,
            default=default_cols,
            help="Each selected column is treated as one mutation row in the heatmap.",
        )

        geo_candidates = _candidate_geography_columns(merged)
        geography_choice = st.selectbox(
            "Population/geography column (optional)",
            options=["None", *geo_candidates],
            index=0,
            help="If selected, a stacked distribution panel is shown for each haplotype.",
        )
        geography_col = None if geography_choice == "None" else geography_choice

        c1, c2 = st.columns(2)
        min_samples = c1.number_input(
            "Minimum samples per haplotype",
            min_value=1,
            max_value=100000,
            value=1,
            step=1,
        )
        max_haplotypes = c2.slider(
            "Max haplotypes shown",
            min_value=10,
            max_value=150,
            value=60,
            step=5,
        )

        render_checkout_haplotype_summary(
            merged_df=merged,
            mutation_columns=selected_mutation_cols,
            geography_column=geography_col,
            min_samples=int(min_samples),
            max_haplotypes=int(max_haplotypes),
        )
