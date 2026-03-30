import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.palettes import Greys256
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.transform import linear_cmap


def _normalise_for_grouping(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df[columns]
        .astype("string")
        .fillna("NA")
        .replace({"<NA>": "NA", "nan": "NA", "None": "NA"})
    )


def _compact_mutation_label(mutation_key: str, alleles: list[str]) -> str:
    position = mutation_key.split("_", 1)[1] if "_" in mutation_key else mutation_key
    valid = [a for a in alleles if a not in ("", "NA")]
    if not valid:
        return position

    parsed = valid
    parsed = [p for p in parsed if isinstance(p, str)]
    m = pd.Series(parsed).str.extract(r"^([A-Za-z*?\-]+)(\d+)([A-Za-z*?\-/]+)$")
    if not m.empty and m.notna().all(axis=1).all():
        refs = m[0].unique().tolist()
        poss = m[1].unique().tolist()
        if len(refs) == 1 and len(poss) == 1:
            ref = refs[0]
            pos = poss[0]
            alts = sorted(m[2].unique().tolist())
            if len(alts) == 1:
                return f"{ref}{pos}{alts[0]}"
            return f"{ref}{pos}[{'/'.join(alts)}]"

    unique_vals = sorted(pd.unique(valid).tolist())
    if len(unique_vals) == 1:
        return str(unique_vals[0])
    return f"{position}[{'/'.join(map(str, unique_vals))}]"


def _extract_position(mutation_key: str) -> str:
    m = pd.Series([mutation_key]).str.extract(r"_(\d+)")
    return m.iloc[0, 0] if not m.empty else ""


def _extract_residue_tokens(value: str) -> list[str]:
    if value in ("", "NA"):
        return []
    tokens = pd.Series([str(value)]).str.extractall(r"([A-Za-z*?\-]+)")[0].tolist()
    return [t for t in tokens if t]


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

    # Matches entries like S108[S/N] and captures reference residue + position.
    bracket_hits = pd.Series([ns_text]).str.extractall(r"([A-Za-z*?\-]+)(\d+)\[[^\]]+\]")
    if not bracket_hits.empty:
        hits = bracket_hits.reset_index(drop=True)
        pos_match = hits[hits[1] == position]
        if not pos_match.empty:
            return str(pos_match.iloc[0, 0])

    # Fallback for simple notations like N51I.
    simple_hits = pd.Series([ns_text]).str.extractall(r"([A-Za-z*?\-]+)(\d+)([A-Za-z*?\-]+)")
    if not simple_hits.empty:
        hits = simple_hits.reset_index(drop=True)
        pos_match = hits[hits[1] == position]
        if not pos_match.empty:
            return str(pos_match.iloc[0, 0])

    return None


def _infer_reference_alleles(merged_df: pd.DataFrame, mutation_columns: list[str]) -> dict[str, str | None]:
    ref_map: dict[str, str | None] = {}
    all_columns = merged_df.columns.tolist()

    for mut_col in mutation_columns:
        position = _extract_position(mut_col)
        if not position:
            ref_map[mut_col] = None
            continue

        ns_col = _find_matching_ns_changes_column(mut_col, all_columns)
        if ns_col is None:
            ref_map[mut_col] = None
            continue

        refs = (
            merged_df[ns_col]
            .astype("string")
            .fillna("NA")
            .map(lambda x: _parse_ref_from_ns_changes(str(x), position))
            .dropna()
        )
        ref_map[mut_col] = refs.mode().iloc[0] if not refs.empty else None

    return ref_map


def _mutation_display_label(mutation_key: str, mode_allele: str, alleles: list[str], ref_allele: str | None) -> str:
    position_match = pd.Series([mutation_key]).str.extract(r"(\d+)")
    position = position_match.iloc[0, 0] if not position_match.empty else mutation_key

    ref = str(ref_allele) if ref_allele not in (None, "", "NA") else (str(mode_allele) if pd.notna(mode_allele) else "")
    if ref in ("", "NA"):
        return _compact_mutation_label(mutation_key, alleles)

    observed = sorted({tok for a in alleles for tok in _extract_residue_tokens(str(a)) if tok not in ("", "NA")})
    if not observed:
        return f"{ref}{position}"
    if len(observed) == 1:
        if observed[0] == ref:
            return f"{ref}{position}"
        return f"{ref}{position}{observed[0]}"

    ordered = [ref] + [x for x in observed if x != ref] if ref in observed else observed
    return f"{ref}{position}[{'/'.join(ordered)}]"


def _hide_x_axis(plt) -> None:
    plt.xaxis.major_label_text_font_size = "0pt"
    plt.xaxis.major_tick_line_color = None
    plt.xaxis.minor_tick_line_color = None
    plt.xgrid.grid_line_color = None


def _prepare_haplotype_summary_data(
    merged_df: pd.DataFrame,
    mutation_columns: list[str],
    min_samples: int,
    max_haplotypes: int,
) -> dict:
    work_cols = ["sample_id"] + mutation_columns

    work = merged_df[work_cols].copy()
    norm_mut = _normalise_for_grouping(work, mutation_columns)
    work[mutation_columns] = norm_mut
    work["haplotype_key"] = norm_mut.agg("|".join, axis=1)

    grouped = (
        work
        .groupby(["haplotype_key", *mutation_columns], dropna=False, sort=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )

    grouped = grouped[grouped["sample_count"] >= min_samples].head(max_haplotypes).reset_index(drop=True)
    if grouped.empty:
        return {"grouped": grouped}

    grouped["haplotype_id"] = [f"H{i + 1}" for i in range(len(grouped))]
    key_map = grouped[["haplotype_key", "haplotype_id"]]

    grouped_sample_rows = work.merge(key_map, how="inner", on="haplotype_key")

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
    mutation_long["is_variant"] = (
        (mutation_long["allele"] != mutation_long["mode_allele"])
        & (~mutation_long["allele"].isin(["NA", ""]))
    ).astype(int)

    order_df = pd.DataFrame({"mutation": mutation_columns})
    alias_pos = order_df["mutation"].str.split("_", n=1, expand=True)
    order_df["alias"] = alias_pos[0]
    order_df["position"] = alias_pos[1].fillna("")
    order_df["pos_num"] = pd.to_numeric(order_df["position"].str.extract(r"(\d+)")[0], errors="coerce")
    order_df["mut_order"] = np.arange(len(order_df))

    mutation_long = mutation_long.merge(order_df, on="mutation", how="left")
    mutation_long = mutation_long.sort_values(["alias", "pos_num", "position", "mut_order"])

    return {
        "grouped": grouped,
        "mutation_long": mutation_long,
    }


def render_checkout_haplotype_summary(
    merged_df: pd.DataFrame,
    mutation_columns: list[str],
    min_samples: int = 1,
    max_haplotypes: int = 60,
) -> None:
    if len(mutation_columns) < 2:
        st.info("Select at least two mutation columns to build a haplotype summary.")
        return

    prep = _prepare_haplotype_summary_data(
        merged_df=merged_df,
        mutation_columns=mutation_columns,
        min_samples=min_samples,
        max_haplotypes=max_haplotypes,
    )
    grouped = prep["grouped"]

    if grouped.empty:
        st.warning("No haplotype combinations pass the current filters.")
        return

    ref_map = _infer_reference_alleles(merged_df, mutation_columns)

    hap_order = grouped["haplotype_id"].tolist()
    shared_x = FactorRange(factors=hap_order)

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

    mutation_long = prep["mutation_long"]
    aliases = mutation_long["alias"].dropna().unique().tolist()

    for alias in aliases:
        sub = mutation_long[mutation_long["alias"] == alias].copy()
        if sub.empty:
            continue

        label_map = {}
        for mut in sub["mutation"].drop_duplicates().tolist():
            rows = sub[sub["mutation"] == mut]
            mode_allele = rows["mode_allele"].iloc[0] if not rows.empty else ""
            alleles = rows["allele"].astype(str).tolist()
            label_map[mut] = _mutation_display_label(mut, mode_allele, alleles, ref_map.get(mut))

        # Ensure categorical factors stay unique, even if different mutation keys
        # would otherwise collapse to the same display label.
        seen = {}
        for mut, lbl in list(label_map.items()):
            k = lbl
            seen[k] = seen.get(k, 0) + 1
            if seen[k] > 1:
                label_map[mut] = f"{lbl} ({seen[k]})"

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
        mapper = linear_cmap(field_name="is_variant", palette=[Greys256[15], Greys256[220]], low=0, high=1)
        hm.rect(
            x="haplotype_id",
            y="mutation_display",
            width=0.92,
            height=0.92,
            source=source,
            line_color="#dddddd",
            fill_color=mapper,
        )
        hm.add_tools(
            HoverTool(
                tooltips=[
                    ("Haplotype", "@haplotype_id"),
                    ("Mutation", "@mutation_display"),
                    ("Mutation key", "@mutation"),
                    ("Allele", "@allele"),
                    ("Modal allele", "@mode_allele"),
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