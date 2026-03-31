import io
import os
import re
import zipfile
import datetime
import pandas as pd
import streamlit as st

from src.haplotype_summary import render_checkout_haplotype_summary

HAPLOTYPES_DIR  = "haplotypes"
_META_CNV_PATH  = "assets/2026-03-31_pf9_meta_cnv_calls.tsv"
_META_COLS      = ["Study", "Country", "Admin level 1", "Year", "Population",
                   "QC pass", "Exclusion reason", "Sample type", "Sample was in Pf8"]
_CNV_COLS       = ["CRT amplification", "GCH1 amplification", "MDR1 amplification",
                   "PM2 PM3 amplification", "HRP2 deletion", "HRP3 deletion"]
_CNV_VALUE_MAP  = {0: "FALSE", 1: "TRUE", -1: "UNDETERMINED"}


@st.cache_data(show_spinner=False)
def _load_meta_cnv() -> pd.DataFrame:
    df = pd.read_csv(_META_CNV_PATH, sep="\t")
    df = df.rename(columns={
        "Sample": "sample_id",
        "CRT_amplification":   "CRT amplification",
        "GCH1_amplification":  "GCH1 amplification",
        "MDR1_amplification":  "MDR1 amplification",
        "PM2_PM3_amplification": "PM2 PM3 amplification",
        "HRP2_deletion":       "HRP2 deletion",
        "HRP3_deletion":       "HRP3 deletion",
    })
    for col in _CNV_COLS:
        if col in df.columns:
            df[col] = df[col].map(_CNV_VALUE_MAP).fillna(df[col])
    return df

# ── Filename parsing ──────────────────────────────────────────────────────────
# Expected pattern: {loci}__{coord_type}__{format_mode}__{YYYYMMDD_HHMMSS}.tsv
# Double-underscore is the field separator; gene names may contain single underscores.

_FNAME_RE = re.compile(
    r'^(?P<loci>.+?)__(?P<coord_type>aa|nt|mixed)__(?P<fmt>default|skip|collapse|wide|expand)__'
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


def _rename_columns_for_coord_type(df: pd.DataFrame, parsed: dict) -> pd.DataFrame:
    """Rename old-style TSV columns to embed the coord type.

    Old-style                              → New-style
    dhps_436.437.540_haplotype            → dhps_aa_436_437_540
    dhps_436.437.540_ns_changes           → dhps_aa_ns_changes
    dhps_436                              → dhps_aa_436
    primer1_20000-20020_haplotype         → primer1_nt_20000_20020
    primer1_20000_20020  (NT interval)    → primer1_nt_20000_20020
    """
    coord_type = parsed.get("type", "?")
    if coord_type not in ("aa", "nt", "mixed"):
        return df

    rename: dict[str, str] = {}
    for col in df.columns:
        if col == "sample_id":
            continue
        # Skip columns already using the new scheme
        if re.search(r"_(aa|nt|mixed)_", col):
            continue

        # {alias}_{pos_sum}_haplotype  →  {alias}_{ct}_{pos_sum_us}
        m = re.match(r"^(.+?)_([\d][\d.\-]*)_haplotype$", col)
        if m:
            pos_str = re.sub(r"[.\-]", "_", m.group(2))
            rename[col] = f"{m.group(1)}_{coord_type}_{pos_str}"
            continue

        # {alias}_{pos_sum}_ns_changes  →  {alias}_{ct}_ns_changes
        m = re.match(r"^(.+?)_([\d][\d.\-]*)_ns_changes$", col)
        if m:
            rename[col] = f"{m.group(1)}_{coord_type}_ns_changes"
            continue

        # NT per-interval range: {alias}_{start}_{end}  →  {alias}_{ct}_{start}_{end}
        if coord_type in ("nt", "mixed"):
            m = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)_(\d+)$", col)
            if m:
                rename[col] = f"{m.group(1)}_{coord_type}_{m.group(2)}_{m.group(3)}"
                continue

        # Single per-position AA: {alias}_{pos}  →  {alias}_{ct}_{pos}
        m = re.match(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)$", col)
        if m:
            rename[col] = f"{m.group(1)}_{coord_type}_{m.group(2)}"
            continue

    return df.rename(columns=rename)


def _is_mutation_column(col_name: str) -> bool:
    if col_name == "sample_id":
        return False
    # Old-style haplotype / ns_changes
    if col_name.endswith("_haplotype") or col_name.endswith("_ns_changes"):
        return False
    # New-style multi-position haplotype: gene_aa_436_437 or gene_nt_2000_2020
    base = col_name.split("__")[0]
    if re.search(r"_(aa|nt|mixed)_\d+_\d+", base):
        return False
    # New-style ns_changes: gene_aa_ns_changes
    if re.search(r"_(aa|nt|mixed)_ns_changes", col_name):
        return False
    if col_name.startswith("Unnamed"):
        return False
    if re.search(r"_(\d+)(?:_\d+)?(?:__.*)?$", col_name):
        return True
    return False


# ── Page ─────────────────────────────────────────────────────────────────────

@st.fragment(run_every=2)
def _poll_haplotypes_dir() -> None:
    """Trigger a full rerun whenever the haplotypes directory gains or loses files."""
    os.makedirs(HAPLOTYPES_DIR, exist_ok=True)
    current = sorted(f for f in os.listdir(HAPLOTYPES_DIR) if f.endswith(".tsv"))
    if current != st.session_state.get("checkout_tsv_snapshot"):
        st.session_state["checkout_tsv_snapshot"] = current
        st.rerun()


def render():
    st.title("Checkout", anchor=False)
    st.caption("Browse saved haplotype files, select any combination, and download a single merged TSV.")

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload TSV", type=["tsv"], label_visibility="collapsed")
    if uploaded is not None:
        dest = os.path.join(HAPLOTYPES_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved `{uploaded.name}`.")

    _poll_haplotypes_dir()

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
        st.caption("Select one or more files above to preview and download.")
        return

    st.divider()

    n_sel = len(selected_files)

    # ── Merge immediately on selection ────────────────────────────────────────
    _merge_cache_key = tuple(sorted(selected_files))
    if st.session_state.get("checkout_merge_key") != _merge_cache_key:
        dfs = []
        load_errors = []
        for fname in selected_files:
            path = os.path.join(HAPLOTYPES_DIR, fname)
            try:
                df = pd.read_csv(path, sep="\t", low_memory=False)
                df = _rename_columns_for_coord_type(df, _parse_filename(fname))
                dfs.append(df)
            except Exception as e:
                load_errors.append(f"`{fname}`: {e}")

        for err in load_errors:
            st.error(err)

        if dfs:
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

            # Always join all meta/CNV columns — toggles filter downloads only
            meta_cnv = _load_meta_cnv()
            merged = merged.merge(
                meta_cnv[["sample_id"] + _META_COLS + _CNV_COLS],
                on="sample_id", how="left",
            )

            all_loci = "+".join(
                _safe_col_prefix(files_df.iloc[i]["_loci_raw"]) for i in selected_indices
            )
            ts_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_name = selected_files[0] if n_sel == 1 else f"merged__{all_loci}__{ts_now}.tsv"

            st.session_state["checkout_merged_df"]   = merged
            st.session_state["checkout_merged_name"] = merged_name
            st.session_state["checkout_merge_key"]   = _merge_cache_key

    # ── Preview + downloads ───────────────────────────────────────────────────
    if "checkout_merged_df" in st.session_state:
        merged      = st.session_state["checkout_merged_df"]
        merged_name = st.session_state["checkout_merged_name"]

        hap_cols = [c for c in merged.columns if c not in _META_COLS + _CNV_COLS]

        tog1, tog2, dl1, dl2 = st.columns([1, 1, 2, 2])
        inc_meta = tog1.toggle("Include metadata", value=True)
        inc_cnv  = tog2.toggle("Include CNV calls", value=True)

        # Columns to include based on toggles
        dl_extra = ((_META_COLS if inc_meta else []) + (_CNV_COLS if inc_cnv else []))
        dl_cols  = hap_cols + [c for c in dl_extra if c in merged.columns]
        dl_df    = merged[dl_cols]

        st.caption(
            f"{dl_df.shape[0]:,} samples × {dl_df.shape[1]} columns"
            + (f" merged from {n_sel} files" if n_sel > 1 else "")
        )
        st.dataframe(dl_df, hide_index=True, width="stretch")

        dl1.download_button(
            "Download merged",
            dl_df.to_csv(sep="\t", index=False),
            file_name=merged_name,
            mime="text/tab-separated-values",
            use_container_width=True,
            type="primary",
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in selected_files:
                zf.write(os.path.join(HAPLOTYPES_DIR, fname), fname)
            if dl_extra:
                extra_df = merged[["sample_id"] + [c for c in dl_extra if c in merged.columns]]
                zf.writestr("extra_columns.tsv", extra_df.to_csv(sep="\t", index=False))
        zip_name = f"haplotypes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        dl2.download_button(
            "Download individual files",
            zip_buf.getvalue(),
            file_name=zip_name,
            mime="application/zip",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Haplotype summary")
        st.caption(
            "Select mutation columns (for example `dhps_aa_456`, `dhps_aa_459`, `dhfr_aa_724`) "
            "to build haplotype combinations and visualise them as a haplotype summary view."
        )

        mutation_candidates = [c for c in merged.columns if _is_mutation_column(c)]
        if len(mutation_candidates) < 1:
            st.info("Need at least one mutation-like columns in the merged data to plot a haplotype summary.")
            return

        st.caption("Select columns directly in the table below (multi-column selection enabled).")
        selector_df = merged[mutation_candidates]
        col_selection = st.dataframe(
            selector_df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="multi-column",
            key="checkout_mutation_column_selector",
        )
        raw_selected_cols = list(col_selection.selection.columns)
        if raw_selected_cols and isinstance(raw_selected_cols[0], int):
            selected_mutation_cols = [selector_df.columns[i] for i in raw_selected_cols if i < len(selector_df.columns)]
        else:
            selected_mutation_cols = raw_selected_cols
        if not selected_mutation_cols:
            st.info("Select two or more mutation columns in the table above to render the haplotype summary.")
            return
        if len(selected_mutation_cols) < 2:
            st.info("Select at least two mutation columns in the table above.")
            return

        c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
        min_samples = c1.number_input(
            "Minimum samples per haplotype",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
        )
        max_haplotypes = c2.slider(
            "Max haplotypes shown",
            min_value=5,
            max_value=100,
            value=15,
            step=1,
        )
        het_mode = c3.radio(
            "Het positions",
            ["Exclude", "Collapse", "Expand"],
            horizontal=True,
        )
        exclude_bad = c4.checkbox(
            "Exclude samples with missing, stop codons, or spanning deletions",
            value = True
        )

        available_meta = [c for c in _META_COLS + _CNV_COLS if c in merged.columns]
        meta_col_raw = st.selectbox(
            "Show distribution by",
            ["(none)"] + available_meta,
            format_func=lambda x: "— none —" if x == "(none)" else x,
        )
        meta_col = None if meta_col_raw == "(none)" else meta_col_raw

        render_checkout_haplotype_summary(
            merged_df=merged,
            mutation_columns=selected_mutation_cols,
            min_samples=int(min_samples),
            max_haplotypes=int(max_haplotypes),
            het_mode=het_mode,
            exclude_bad=exclude_bad,
            meta_col=meta_col,
        )
