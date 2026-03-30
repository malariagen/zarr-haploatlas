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
        mutation_order = (
            sub[["mutation", "pos_num", "position", "mut_order"]]
            .drop_duplicates()
            .sort_values(["pos_num", "position", "mut_order"])
        )
        y_order = mutation_order["mutation"].tolist()[::-1]

        tick_labels = {}
        for mut in y_order:
            alleles = sub.loc[sub["mutation"] == mut, "allele"].astype(str).tolist()
            tick_labels[mut] = _compact_mutation_label(mut, alleles)

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
            y="mutation",
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
                    ("Mutation", "@mutation"),
                    ("Allele", "@allele"),
                    ("Modal allele", "@mode_allele"),
                    ("Samples", "@sample_count"),
                ]
            )
        )
        hm.yaxis.axis_label = alias
        hm.yaxis.major_label_overrides = tick_labels
        hm.ygrid.grid_line_color = None
        _hide_x_axis(hm)

        figures.append(hm)

    cn_layout = column(*figures, sizing_mode="stretch_width")
    total_height = int(sum((fig.height or 260) for fig in figures) + 40)
    components.html(file_html(cn_layout, CDN), height=total_height, scrolling=True)