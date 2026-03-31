import re
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.palettes import Category20, Greys256
from bokeh.plotting import figure
from bokeh.resources import CDN


_POS_RE = re.compile(r"_(\d+)")
_RESIDUE_TOKEN_RE = re.compile(r"([A-Za-z*?\-]+)")
_BRACKET_REF_RE = re.compile(r"([A-Za-z*?\-]+)(\d+)\[[^\]]+\]")
_SIMPLE_REF_RE = re.compile(r"([A-Za-z*?\-]+)(\d+)([A-Za-z*?\-]+)")

_GREY_CELL = "#D3D3D3"
_BLUE_CELL = "#4878CF"

_POPULATION_COLOURS: dict[str, str] = {
    "LA-W":    "#b8e186",
    "LA-E":    "#4dac26",
    "AF-W":    "#e31a1c",
    "AF-C":    "#fd8d3c",
    "AF-NE":   "#bb8129",
    "AF-E":    "#fecc5c",
    "AS-S-E":  "#dfc0eb",
    "AS-S-FE": "#984ea3",
    "AS-SE-W": "#9ecae1",
    "AS-SE-E": "#3182bd",
    "AS-SE-M": "#02818a",
    "OC-NG":   "#f781bf",
    "EU":      "#003399",
}

# ── Allele classification helpers ─────────────────────────────────────────────

def _is_het_allele(allele: str) -> bool:
    """Detect het alleles in any format produced by haplotype generation.

    Per-position columns use:
      "A,S"  – ordered het in default/expand mode (major,minor)
      "#"    – unresolved het in skip mode
    Haplotype columns additionally use "[A/S]" but those aren't mutation cols.
    """
    s = str(allele)
    return "#" in s or "/" in s or "|" in s or "," in s


def _is_bad_allele(allele: str) -> bool:
    s = str(allele)
    return s in ("NA", "*", "!", "-", "?", "", "nan", "<NA>", "None") or s.startswith("<NA>")


def _clean_alleles(alleles: list[str]) -> list[str]:
    """Return only unambiguous, non-missing alleles."""
    return [a for a in alleles if not _is_bad_allele(a) and not _is_het_allele(a)]


# ── Core helpers ──────────────────────────────────────────────────────────────

def _normalise_for_grouping(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df[columns]
        .astype("string")
        .fillna("NA")
        .replace({"<NA>": "NA", "nan": "NA", "None": "NA"})
    )


def _extract_position(mutation_key: str) -> str:
    m = _POS_RE.search(mutation_key)
    return m.group(1) if m else ""


def _extract_residue_tokens(value: str) -> list[str]:
    if value in ("", "NA"):
        return []
    return [t for t in _RESIDUE_TOKEN_RE.findall(str(value)) if t]


def _compact_mutation_label(mutation_key: str, alleles: list[str]) -> str:
    position = _extract_position(mutation_key) or mutation_key
    valid = _clean_alleles(alleles)
    if not valid:
        return position
    m = pd.Series(valid).str.extract(r"^([A-Za-z*?\-]+)(\d+)([A-Za-z*?\-/]+)$")
    if not m.empty and m.notna().all(axis=1).all():
        refs = m[0].unique().tolist()
        poss = m[1].unique().tolist()
        if len(refs) == 1 and len(poss) == 1:
            alts = sorted(m[2].unique().tolist())
            if len(alts) == 1:
                return f"{refs[0]}{poss[0]}{alts[0]}"
            return f"{refs[0]}{poss[0]}[{'/'.join(alts)}]"
    unique_vals = sorted(pd.unique(valid).tolist())
    if len(unique_vals) == 1:
        return str(unique_vals[0])
    return f"{position}[{'/'.join(map(str, unique_vals))}]"


def _mutation_display_label(
    mutation_key: str, mode_allele: str, alleles: list[str], ref_allele: str | None
) -> str:
    """Return a clean label like S436A, using only unambiguous alleles."""
    position = _extract_position(mutation_key) or mutation_key
    ref = (
        str(ref_allele)
        if ref_allele not in (None, "", "NA")
        else (str(mode_allele) if pd.notna(mode_allele) else "")
    )
    if ref in ("", "NA"):
        return _compact_mutation_label(mutation_key, alleles)

    clean = _clean_alleles(alleles)
    non_ref_tokens = [
        tok
        for a in clean
        for tok in _extract_residue_tokens(str(a))
        if tok not in ("", "NA", ref)
    ]
    if not non_ref_tokens:
        return f"{ref}{position}"
    most_common_alt = Counter(non_ref_tokens).most_common(1)[0][0]
    return f"{ref}{position}{most_common_alt}"


def _find_matching_ns_changes_column(mutation_col: str, all_columns: list[str]) -> str | None:
    alias = mutation_col.split("_", 1)[0]
    suffix = mutation_col.split("__", 1)[1] if "__" in mutation_col else None
    candidates = [
        c for c in all_columns
        if c.startswith(f"{alias}_") and "_ns_changes" in c
    ]
    if not candidates:
        return None
    if suffix:
        suffix_matches = [c for c in candidates if c.endswith(f"__{suffix}")]
        if suffix_matches:
            return suffix_matches[0]
    return candidates[0]


def _parse_ref_from_ns_changes(ns_text: str, position: str) -> str | None:
    if not ns_text or ns_text in ("NA", ""):
        return None
    for ref, pos in _BRACKET_REF_RE.findall(ns_text):
        if pos == position:
            return ref
    for ref, pos, _alt in _SIMPLE_REF_RE.findall(ns_text):
        if pos == position:
            return ref
    return None


def _infer_reference_alleles(
    merged_df: pd.DataFrame, mutation_columns: list[str]
) -> dict[str, str | None]:
    """Infer reference allele per mutation column.

    Primary: samples where ns_changes is empty/None are wildtype; their allele
    at that position IS the reference.
    Fallback: parse the ref letter from mutation notation in ns_changes text.
    """
    ref_map: dict[str, str | None] = {}
    all_columns = merged_df.columns.tolist()

    for mut_col in mutation_columns:
        ns_col = _find_matching_ns_changes_column(mut_col, all_columns)

        if ns_col is not None:
            wt_mask = merged_df[ns_col].isna() | merged_df[ns_col].astype("string").isin(
                ["NA", "", "None", "nan", "<NA>"]
            )
            wt_alleles = (
                merged_df.loc[wt_mask, mut_col]
                .astype("string")
                .replace({"NA": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA, "": pd.NA})
                .dropna()
            )
            clean = _clean_alleles(wt_alleles.tolist())
            if clean:
                ref_map[mut_col] = pd.Series(clean).mode().iloc[0]
                continue

            # Fallback: parse ref letter from mutation notation
            position = _extract_position(mut_col)
            if position:
                refs = (
                    merged_df[ns_col]
                    .astype("string")
                    .fillna("NA")
                    .map(lambda x: _parse_ref_from_ns_changes(str(x), position))
                    .dropna()
                )
                if not refs.empty:
                    ref_map[mut_col] = refs.mode().iloc[0]
                    continue

        ref_map[mut_col] = None

    return ref_map


def _cat_palette(n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= 20:
        return list(Category20[max(n, 3)])[:n]
    base = list(Category20[20])
    return (base * (n // 20 + 1))[:n]


def _safe_bokeh_field(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not safe or safe[0].isdigit():
        safe = f"cat_{safe}"
    return safe


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _hide_x_axis(plt) -> None:
    plt.xaxis.major_label_text_font_size = "0pt"
    plt.xaxis.major_tick_line_color = None
    plt.xaxis.minor_tick_line_color = None
    plt.xgrid.grid_line_color = None


# ── Data preparation ──────────────────────────────────────────────────────────

def _prepare_haplotype_summary_data(
    merged_df: pd.DataFrame,
    mutation_columns: list[str],
    min_samples: int,
    max_haplotypes: int,
    het_mode: str = "Exclude",
    exclude_bad: bool = False,
    ref_map: dict | None = None,
) -> dict:
    work_cols = ["sample_id"] + mutation_columns
    work = merged_df[work_cols].copy()
    norm_mut = _normalise_for_grouping(work, mutation_columns)
    work[mutation_columns] = norm_mut

    # ── Het handling ───────────────────────────────────────────────────────────
    if het_mode == "Exclude":
        het_mask = work[mutation_columns].apply(lambda col: col.map(_is_het_allele)).any(axis=1)
        work = work[~het_mask].reset_index(drop=True)

    elif het_mode == "Collapse":
        for col in mutation_columns:
            ref = (ref_map or {}).get(col)
            het_mask = work[col].map(_is_het_allele)
            work.loc[het_mask, col] = ref if ref else "NA"

    elif het_mode == "Expand":
        rows_out = []
        for _, row in work.iterrows():
            if not any(_is_het_allele(row[c]) for c in mutation_columns):
                rows_out.append(row.to_dict())
                continue
            r1, r2 = row.to_dict(), row.to_dict()
            for col in mutation_columns:
                v = str(row[col])
                if "," in v:
                    # "A,S" — major,minor from default/expand mode
                    parts = [p.strip() for p in v.split(",", 1)]
                    r1[col] = parts[0]
                    r2[col] = parts[1]
                elif "/" in v:
                    # "[A/S]" or "A/S"
                    parts = [p.strip().strip("[]") for p in v.split("/", 1)]
                    r1[col] = parts[0]
                    r2[col] = parts[1]
                elif "#" in v:
                    # Unresolvable het — collapse both rows to ref
                    ref = (ref_map or {}).get(col)
                    r1[col] = ref if ref else "NA"
                    r2[col] = ref if ref else "NA"
            rows_out.append(r1)
            rows_out.append(r2)
        work = pd.DataFrame(rows_out).reset_index(drop=True)

    # ── Exclude missing / stop / spanning deletion ─────────────────────────────
    if exclude_bad:
        bad_mask = work[mutation_columns].apply(lambda col: col.map(_is_bad_allele)).any(axis=1)
        work = work[~bad_mask].reset_index(drop=True)

    norm_mut2 = _normalise_for_grouping(work, mutation_columns)
    work[mutation_columns] = norm_mut2
    work["haplotype_key"] = norm_mut2.agg("|".join, axis=1)

    # Group solely by haplotype_key so that haplotypes made identical by het
    # handling (e.g. Collapse) are merged into one bar.  Including mutation
    # columns as additional groupby keys risks splitting what should be a single
    # haplotype when pandas StringDtype NA representations differ slightly.
    hap_counts = (
        work
        .groupby("haplotype_key", dropna=False, sort=False)
        .size()
        .reset_index(name="sample_count")
    )
    hap_alleles = (
        work
        .groupby("haplotype_key", dropna=False, sort=False)[mutation_columns]
        .first()
        .reset_index()
    )
    grouped = (
        hap_counts
        .merge(hap_alleles, on="haplotype_key")
        .sort_values("sample_count", ascending=False)
    )
    grouped = grouped[grouped["sample_count"] >= min_samples].head(max_haplotypes).reset_index(drop=True)
    if grouped.empty:
        return {"grouped": grouped}

    grouped["haplotype_id"] = [f"H{i + 1}" for i in range(len(grouped))]
    key_map = grouped[["haplotype_key", "haplotype_id"]]
    grouped_sample_rows = work.merge(key_map, how="inner", on="haplotype_key")

    # Mode allele — kept as fallback for label generation when ref unknown
    mode_map = {}
    for col in mutation_columns:
        valid = grouped_sample_rows[col][~grouped_sample_rows[col].isin(["NA", ""])]
        mode_map[col] = valid.mode().iloc[0] if not valid.empty else "NA"

    mutation_long = (
        grouped[["haplotype_id", "sample_count", *mutation_columns]]
        .melt(
            id_vars=["haplotype_id", "sample_count"],
            value_vars=mutation_columns,
            var_name="mutation",
            value_name="allele",
        )
    )

    mode_lookup = pd.Series(mode_map, name="mode_allele").rename_axis("mutation").reset_index()
    mutation_long = mutation_long.merge(mode_lookup, on="mutation", how="left")

    _ref = ref_map or {}
    ref_lookup = pd.Series(
        {k: (v if v is not None else "NA") for k, v in _ref.items()},
        name="ref_allele",
    ).rename_axis("mutation").reset_index()
    if not ref_lookup.empty:
        mutation_long = mutation_long.merge(ref_lookup, on="mutation", how="left")
    else:
        mutation_long["ref_allele"] = "NA"
    mutation_long["ref_allele"] = mutation_long["ref_allele"].fillna("NA")

    # is_mutant: 1 = non-ref clean allele → blue; 0 = ref / missing / het → grey
    def _is_mutant(row) -> int:
        ref = str(row["ref_allele"])
        al = str(row["allele"])
        if ref in ("NA", "", "None", "nan"):
            return 0
        if al in ("NA", "", "None", "nan"):
            return 0
        if _is_het_allele(al) or _is_bad_allele(al):
            return 0
        return int(al != ref)

    mutation_long["is_mutant"] = mutation_long.apply(_is_mutant, axis=1)
    mutation_long["cell_color"] = mutation_long["is_mutant"].map({0: _GREY_CELL, 1: _BLUE_CELL})

    order_df = pd.DataFrame({"mutation": mutation_columns})
    alias_pos = order_df["mutation"].str.split("_", n=1, expand=True)
    order_df["alias"] = alias_pos[0]
    order_df["position"] = alias_pos[1].fillna("")
    order_df["pos_num"] = pd.to_numeric(
        order_df["position"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    order_df["mut_order"] = np.arange(len(order_df))

    mutation_long = mutation_long.merge(order_df, on="mutation", how="left")
    mutation_long = mutation_long.sort_values(["alias", "pos_num", "position", "mut_order"])

    return {
        "grouped": grouped,
        "mutation_long": mutation_long,
        "grouped_sample_rows": grouped_sample_rows,
    }


# ── Public render function ────────────────────────────────────────────────────

def render_checkout_haplotype_summary(
    merged_df: pd.DataFrame,
    mutation_columns: list[str],
    min_samples: int = 1,
    max_haplotypes: int = 60,
    het_mode: str = "Exclude",
    exclude_bad: bool = False,
    meta_col: str | None = None,
) -> None:
    if len(mutation_columns) < 2:
        st.info("Select at least two mutation columns to build a haplotype summary.")
        return

    ref_map = _infer_reference_alleles(merged_df, mutation_columns)

    prep = _prepare_haplotype_summary_data(
        merged_df=merged_df,
        mutation_columns=mutation_columns,
        min_samples=min_samples,
        max_haplotypes=max_haplotypes,
        het_mode=het_mode,
        exclude_bad=exclude_bad,
        ref_map=ref_map,
    )
    grouped = prep["grouped"]

    if grouped.empty:
        st.warning("No haplotype combinations pass the current filters.")
        return

    hap_order = grouped["haplotype_id"].tolist()
    shared_x = FactorRange(factors=hap_order)

    # ── Sample count bar ───────────────────────────────────────────────────────
    count_src = ColumnDataSource(grouped[["haplotype_id", "sample_count"]])
    count_fig = figure(
        x_range=shared_x,
        height=280,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,reset,save",
        toolbar_location="right",
    )
    count_fig.vbar(x="haplotype_id", top="sample_count", width=0.85, source=count_src, color="#5C80BC")
    count_fig.add_tools(HoverTool(tooltips=[("Haplotype", "@haplotype_id"), ("Samples", "@sample_count")]))
    count_fig.yaxis.axis_label = "Number of samples"
    _hide_x_axis(count_fig)

    figures = [count_fig]

    # ── Metadata distribution chart ────────────────────────────────────────────
    if meta_col and meta_col in merged_df.columns and "grouped_sample_rows" in prep:
        gsr = prep["grouped_sample_rows"][["sample_id", "haplotype_id"]]
        meta_vals = merged_df[["sample_id", meta_col]].copy()
        meta_vals[meta_col] = meta_vals[meta_col].fillna("Unknown").astype(str)
        gsr_meta = gsr.merge(meta_vals, on="sample_id", how="left")
        gsr_meta[meta_col] = gsr_meta[meta_col].fillna("Unknown")

        dist = (
            gsr_meta.groupby(["haplotype_id", meta_col])
            .size()
            .reset_index(name="cnt")
        )
        totals = dist.groupby("haplotype_id")["cnt"].sum().rename("total")
        dist = dist.join(totals, on="haplotype_id")
        dist["pct"] = dist["cnt"] / dist["total"] * 100

        # Cap at top-20 categories by overall count
        top_cats = (
            dist.groupby(meta_col)["cnt"].sum()
            .nlargest(20)
            .index.tolist()
        )
        dist.loc[~dist[meta_col].isin(top_cats), meta_col] = "Other"
        dist = dist.groupby(["haplotype_id", meta_col])["pct"].sum().reset_index()

        categories = sorted(dist[meta_col].unique().tolist())

        dist_wide = (
            dist.pivot(index="haplotype_id", columns=meta_col, values="pct")
            .fillna(0)
            .reindex(hap_order)
            .fillna(0)
            .reset_index()
        )

        # Sanitise category names to valid Bokeh field names
        used: dict[str, int] = {}
        safe_map: dict[str, str] = {}
        for cat in categories:
            s = _safe_bokeh_field(cat)
            cnt = used.get(s, 0)
            used[s] = cnt + 1
            safe_map[cat] = f"{s}_{cnt}" if cnt else s

        dist_wide = dist_wide.rename(columns=safe_map)
        safe_stackers = [safe_map[c] for c in categories]
        if meta_col == "Population":
            palette = [_POPULATION_COLOURS.get(c, "#aaaaaa") for c in categories]
        else:
            palette = _cat_palette(len(categories))

        meta_src = ColumnDataSource(dist_wide)
        meta_fig = figure(
            x_range=shared_x,
            height=200,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="right",
        )
        meta_fig.vbar_stack(
            stackers=safe_stackers,
            x="haplotype_id",
            width=0.85,
            color=palette,
            source=meta_src,
            legend_label=categories,
        )
        meta_fig.yaxis.axis_label = f"{meta_col} (%)"
        meta_fig.y_range.end = 100
        meta_fig.legend.location = "top_right"
        meta_fig.legend.label_text_font_size = "9pt"
        meta_fig.legend.spacing = 1
        if len(categories) > 12:
            meta_fig.legend.visible = False
        _hide_x_axis(meta_fig)
        figures.append(meta_fig)

    # ── Heatmap figures (one per gene alias) ───────────────────────────────────
    mutation_long = prep["mutation_long"]
    aliases = mutation_long["alias"].dropna().unique().tolist()

    for alias in aliases:
        sub = mutation_long[mutation_long["alias"] == alias].copy()
        if sub.empty:
            continue

        meta_agg = (
            sub.groupby("mutation", sort=False)
            .agg(
                mode_allele=("mode_allele", "first"),
                ref_allele=("ref_allele", "first"),
                alleles=("allele", lambda s: [str(v) for v in s.tolist()]),
            )
            .reset_index()
        )

        label_map: dict[str, str] = {}
        for _, row in meta_agg.iterrows():
            label_map[row["mutation"]] = _mutation_display_label(
                row["mutation"],
                row["mode_allele"],
                _clean_alleles(row["alleles"]),
                row["ref_allele"],
            )

        # Deduplicate labels that would collapse to the same string
        seen: dict[str, int] = {}
        for mut, lbl in list(label_map.items()):
            seen[lbl] = seen.get(lbl, 0) + 1
            if seen[lbl] > 1:
                label_map[mut] = f"{lbl} ({seen[lbl]})"

        sub["mutation_display"] = sub["mutation"].map(label_map)

        mutation_order = (
            sub[["mutation", "pos_num", "position", "mut_order"]]
            .drop_duplicates()
            .sort_values(["pos_num", "position", "mut_order"])
        )
        y_order = [label_map[m] for m in mutation_order["mutation"].tolist()][::-1]

        source = ColumnDataSource(sub)
        hm = figure(
            x_range=shared_x,
            y_range=y_order,
            height=max(220, 35 * len(y_order)),
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,reset,save",
            toolbar_location="right",
        )
        hm.rect(
            x="haplotype_id",
            y="mutation_display",
            width=0.92,
            height=0.92,
            source=source,
            line_color="#dddddd",
            fill_color="cell_color",
        )
        hm.add_tools(
            HoverTool(
                tooltips=[
                    ("Haplotype", "@haplotype_id"),
                    ("Position", "@mutation_display"),
                    ("Allele", "@allele"),
                    ("Reference", "@ref_allele"),
                    ("Samples", "@sample_count"),
                ]
            )
        )
        hm.yaxis.axis_label = alias
        hm.ygrid.grid_line_color = None
        _hide_x_axis(hm)

        figures.append(hm)

    cn_layout = column(*figures, sizing_mode="stretch_width")
    total_height = int(sum((fig.height or 260) for fig in figures) + 40)
    components.html(file_html(cn_layout, CDN), height=total_height, scrolling=True)
