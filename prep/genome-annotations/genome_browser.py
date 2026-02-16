"""
Bokeh Genome Browser
====================
A genome browser function that displays genomic features as a Gantt-chart
and annotates CDS features with per-codon amino acid mappings, including
careful handling of amino acids whose codons are split across exons.

Usage:
    from genome_browser import genome_browser
    genome_browser(
        locus="Pf3D7_07_v3:412,321-453,215",
        filtered_df=filtered,
        genome_fasta_path="PlasmoDB-54_Pfalciparum3D7_Genome.fasta",
        protein_fasta_path="PlasmoDB-68_Pfalciparum3D7_AnnotatedProteins.fasta"
    )
"""

import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, HoverTool, Range1d, Label, LabelSet,
    CustomJS, Title, Whisker, Band, Span, BoxAnnotation, Arrow,
    NormalHead, OpenHead, VeeHead,
    WheelZoomTool, PanTool, BoxZoomTool, ResetTool, SaveTool, CrosshairTool,
)
from bokeh.io import output_notebook
from collections import defaultdict


# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

AA_COLORS = {
    "A": "#8CBA80", "V": "#8CBA80", "I": "#65A34D", "L": "#65A34D", "M": "#4E8C38",
    "F": "#6A9FD6", "W": "#3B7DC4", "Y": "#5490CE",
    "R": "#E06060", "H": "#E88888", "K": "#D04040",
    "D": "#C77CDB", "E": "#B55FCC",
    "S": "#F5C842", "T": "#E8B820", "N": "#F0D060", "Q": "#DDA810",
    "C": "#F5A623", "G": "#CCCCCC", "P": "#AAAAAA",
    "*": "#444444", "X": "#999999",
}

FEATURE_COLORS = {
    "gene":             "#4477AA",
    "mRNA":             "#66CCEE",
    "rRNA":             "#228833",
    "tRNA":             "#CCBB44",
    "ncRNA":            "#EE6677",
    "snRNA":            "#AA3377",
    "snoRNA":           "#CC6688",
    "exon":             "#88CCAA",
    "five_prime_UTR":   "#BBBBBB",
    "three_prime_UTR":  "#999999",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_locus(locus_str: str):
    """Parse a locus string like 'Pf3D7_07_v3:412,321-453,215'."""
    m = re.match(r"(.+?):([0-9,]+)\s*-\s*([0-9,]+)", locus_str)
    if not m:
        raise ValueError(
            f"Invalid locus format: '{locus_str}'. "
            "Expected something like 'Pf3D7_07_v3:412,321-453,215'"
        )
    chrom = m.group(1)
    start = int(m.group(2).replace(",", ""))
    end = int(m.group(3).replace(",", ""))
    return chrom, start, end


def load_chromosome_seq(fasta_path: str, chrom: str) -> str:
    """Return the full sequence string for *chrom* (upper-cased)."""
    for rec in SeqIO.parse(fasta_path, "fasta"):
        if rec.id == chrom:
            return str(rec.seq).upper()
    raise ValueError(f"Chromosome '{chrom}' not found in {fasta_path}")


def load_protein_seqs(fasta_path: str) -> dict:
    """Return {gene_id: protein_sequence} from the annotated-proteins FASTA."""
    proteins: dict[str, str] = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        m = re.search(r"gene=(\S+)", rec.description)
        if m:
            gid = m.group(1)
            if gid not in proteins:
                proteins[gid] = str(rec.seq)
    return proteins


# ---------------------------------------------------------------------------
# Codon ↔ genomic-coordinate mapping
# ---------------------------------------------------------------------------

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _coding_positions(gene_cds_df: pd.DataFrame, strand: str) -> list[int]:
    """
    Return a list of 1-based genomic positions in mRNA 5'→3' reading order.
    """
    if strand == "+":
        exons = gene_cds_df.sort_values("start")
    else:
        exons = gene_cds_df.sort_values("end", ascending=False)

    positions: list[int] = []
    for _, row in exons.iterrows():
        s, e = int(row["start"]), int(row["end"])
        if strand == "+":
            positions.extend(range(s, e + 1))
        else:
            positions.extend(range(e, s - 1, -1))
    return positions


def codon_genomic_mapping(
    gene_cds_df: pd.DataFrame,
    chrom_seq: str,
    strand: str,
    protein_seq: str | None = None,
) -> list[dict]:
    """
    Map every amino acid of a gene to its three genomic base positions.
    Split codons across exons are handled naturally.
    """
    positions = _coding_positions(gene_cds_df, strand)

    mappings: list[dict] = []
    for i in range(0, len(positions) - 2, 3):
        triple = positions[i : i + 3]
        if len(triple) < 3:
            break

        if strand == "+":
            codon = "".join(chrom_seq[p - 1] for p in triple)
        else:
            codon = "".join(chrom_seq[p - 1] for p in triple).translate(COMPLEMENT)

        aa_idx = i // 3

        if protein_seq is not None and aa_idx < len(protein_seq):
            aa = protein_seq[aa_idx]
        else:
            aa = str(Seq(codon).translate())

        mappings.append(
            {
                "aa": aa,
                "aa_index": aa_idx,
                "codon": codon,
                "positions": triple,
            }
        )

    return mappings


def contiguous_blocks(positions: list[int]) -> list[list[int]]:
    """
    Split a list of genomic positions into runs of genomically adjacent bases.
    """
    if not positions:
        return []
    sp = sorted(positions)
    blocks: list[list[int]] = [[sp[0]]]
    for i in range(1, len(sp)):
        if sp[i] - sp[i - 1] == 1:
            blocks[-1].append(sp[i])
        else:
            blocks.append([sp[i]])
    return blocks


# ---------------------------------------------------------------------------
# Overlap packing — used for both CDS genes and non-CDS features
# ---------------------------------------------------------------------------

def pack_rows(features: list[dict], gap: int = 0) -> list[list[dict]]:
    """
    Greedily pack features into horizontal rows so that no two features
    in the same row overlap.  Returns the minimum number of rows needed.
    """
    rows: list[list[dict]] = []
    for f in sorted(features, key=lambda x: x["start"]):
        placed = False
        for row in rows:
            if f["start"] > row[-1]["end"] + gap:
                row.append(f)
                placed = True
                break
        if not placed:
            rows.append([f])
    return rows


# ---------------------------------------------------------------------------
# Main genome_browser function
# ---------------------------------------------------------------------------

def genome_browser(
    locus: str,
    filtered_df: pd.DataFrame,
    genome_fasta_path: str,
    protein_fasta_path: str,
    *,
    plot_width: int = 1800,
    plot_height: int = 400,
):
    """
    Render an interactive Bokeh genome browser for the given *locus*.

    Parameters
    ----------
    locus : str
        Genomic window, e.g. ``"Pf3D7_07_v3:412,321-453,215"``.
    filtered_df : DataFrame
        Feature table with columns ``seqid, type, start, end, strand,
        phase, Name, description, gene_id, exon``.
    genome_fasta_path, protein_fasta_path : str
        Paths to the genome and annotated-protein FASTA files.
    plot_width, plot_height : int
        Bokeh figure dimensions in pixels.

    Returns
    -------
    bokeh.plotting.Figure
    """

    # ------------------------------------------------------------------
    # 1. Parse inputs & load data
    # ------------------------------------------------------------------
    chrom, view_start, view_end = parse_locus(locus)
    view_span = view_end - view_start

    chrom_seq = load_chromosome_seq(genome_fasta_path, chrom)
    proteins = load_protein_seqs(protein_fasta_path)

    chrom_df = filtered_df[filtered_df["seqid"] == chrom].copy()
    view_df = chrom_df[(chrom_df["end"] >= view_start) & (chrom_df["start"] <= view_end)]

    cds_view = view_df[view_df["type"] == "CDS"]
    noncds_view = view_df[view_df["type"] != "CDS"]

    # ------------------------------------------------------------------
    # 2. Create the Bokeh figure — only X-axis zoom/pan
    # ------------------------------------------------------------------
    wheel_zoom = WheelZoomTool(dimensions="width")
    p = figure(
        width=plot_width,
        height=plot_height,
        x_range=Range1d(view_start - view_span * 0.01, view_end + view_span * 0.01),
        title=f"{chrom}:{view_start:,}-{view_end:,}",
        x_axis_label="Genomic position (bp)",
        tools=[
            PanTool(dimensions="width"),
            wheel_zoom,
            BoxZoomTool(dimensions="width"),
            ResetTool(),
            SaveTool(),
            CrosshairTool(),
        ],
        active_scroll=wheel_zoom,
        output_backend="webgl",
    )
    p.title.text_font_size = "14pt"
    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xgrid.grid_line_dash = "dotted"
    p.xgrid.grid_line_alpha = 0.4
    p.xaxis.formatter.use_scientific = False
    p.xaxis[0].formatter.use_scientific = False

    # ------------------------------------------------------------------
    # 3. CDS TRACK — pack genes into minimum y-rows
    # ------------------------------------------------------------------
    gene_order = (
        cds_view.groupby("gene_id")["start"]
        .min()
        .sort_values()
        .index.tolist()
    )

    # Build per-gene spans for packing
    gene_spans: list[dict] = []
    for gene_id in gene_order:
        all_exons = chrom_df[
            (chrom_df["type"] == "CDS") & (chrom_df["gene_id"] == gene_id)
        ]
        gene_spans.append({
            "gene_id": gene_id,
            "start": int(all_exons["start"].min()),
            "end":   int(all_exons["end"].max()),
        })

    # Greedy packing — returns minimum number of rows
    cds_packed_rows = pack_rows(gene_spans, gap=500)

    cds_track_y_base = 0.0
    row_height = 1.5
    gene_y_positions: dict[str, float] = {}
    aa_text_renderers = []  # collect text glyphs for conditional visibility

    for ri, row in enumerate(cds_packed_rows):
        gene_y = cds_track_y_base - ri * row_height
        for gspan in row:
            gene_y_positions[gspan["gene_id"]] = gene_y

    # Now draw each gene at its packed Y position
    for gene_id, gene_y in gene_y_positions.items():
        all_exons = chrom_df[
            (chrom_df["type"] == "CDS") & (chrom_df["gene_id"] == gene_id)
        ]
        strand = all_exons.iloc[0]["strand"]
        desc = str(all_exons.iloc[0].get("description", ""))

        # Thin intron line
        gene_left = max(int(all_exons["start"].min()), view_start)
        gene_right = min(int(all_exons["end"].max()), view_end)
        p.segment(
            x0=[gene_left], y0=[gene_y], x1=[gene_right], y1=[gene_y],
            line_width=1.5, line_color="#555555",
        )

        # Exon background blocks
        exons_in_view = all_exons[
            (all_exons["end"] >= view_start) & (all_exons["start"] <= view_end)
        ]
        for _, exon_row in exons_in_view.iterrows():
            p.quad(
                left=max(int(exon_row["start"]), view_start) - 0.5,
                right=min(int(exon_row["end"]), view_end) + 0.5,
                top=gene_y + 0.45,
                bottom=gene_y - 0.45,
                fill_color="#E8E8E8",
                fill_alpha=0.35,
                line_color="#888888",
                line_width=0.8,
            )

        # Amino-acid codon mapping
        protein_seq = proteins.get(gene_id)
        aa_maps = codon_genomic_mapping(all_exons, chrom_seq, strand, protein_seq)

        # Collect glyph data for codon blocks
        q_left, q_right, q_top, q_bottom = [], [], [], []
        q_color, q_aa, q_idx, q_codon = [], [], [], []
        # Collect label data — one per amino acid (placed on largest block)
        lbl_x, lbl_y, lbl_text = [], [], []

        for am in aa_maps:
            blocks = contiguous_blocks(am["positions"])
            aa = am["aa"]
            color = AA_COLORS.get(aa, "#CCCCCC")

            largest_block = max(blocks, key=len)
            largest_mid = (min(largest_block) + max(largest_block)) / 2.0

            for blk in blocks:
                bmin, bmax = min(blk), max(blk)
                if bmax < view_start or bmin > view_end:
                    continue

                q_left.append(bmin - 0.5)
                q_right.append(bmax + 0.5)
                q_top.append(gene_y + 0.4)
                q_bottom.append(gene_y - 0.4)
                q_color.append(color)
                q_aa.append(aa)
                q_idx.append(am["aa_index"] + 1)
                q_codon.append(am["codon"])

            # Place label on the largest block (if in view)
            lb_min, lb_max = min(largest_block), max(largest_block)
            if lb_max >= view_start and lb_min <= view_end:
                lbl_x.append(largest_mid)
                lbl_y.append(gene_y)
                lbl_text.append(aa)

        # Draw the codon quads
        if q_left:
            src = ColumnDataSource(
                data=dict(
                    left=q_left, right=q_right, top=q_top, bottom=q_bottom,
                    fill_color=q_color, aa=q_aa, aa_idx=q_idx, codon=q_codon,
                )
            )
            rend = p.quad(
                left="left", right="right", top="top", bottom="bottom",
                fill_color="fill_color", line_color="white", line_width=0.3,
                fill_alpha=0.85, source=src,
            )
            p.add_tools(
                HoverTool(
                    renderers=[rend],
                    tooltips=[
                        ("Amino acid", "@aa"),
                        ("Residue #", "@aa_idx"),
                        ("Codon", "@codon"),
                        ("Gene", gene_id),
                        ("Strand", strand),
                    ],
                )
            )

        # Amino-acid single-letter labels — rendered inside the codon
        # rectangles; visibility toggled by a JS callback on zoom level
        if lbl_x:
            lsrc = ColumnDataSource(data=dict(x=lbl_x, y=lbl_y, text=lbl_text))
            txt_rend = p.text(
                x="x", y="y", text="text", source=lsrc,
                text_font_size="7pt", text_align="center",
                text_baseline="middle", text_color="black",
                text_font_style="bold",
            )
            txt_rend.visible = (view_span <= 3000)
            aa_text_renderers.append(txt_rend)

        # Gene ID label
        strand_arrow = "\u2192" if strand == "+" else "\u2190"
        p.add_layout(
            Label(
                x=gene_left, y=gene_y + 0.55,
                text=f"{gene_id} {strand_arrow}  {desc[:60]}",
                text_font_size="8pt", text_color="#333366",
            )
        )

    # --- JS callback: show AA letters only when zoomed in enough ---
    if aa_text_renderers:
        callback = CustomJS(
            args=dict(renderers=aa_text_renderers),
            code="""
                const span = cb_obj.end - cb_obj.start;
                const visible = span <= 3000;
                for (const r of renderers) {
                    r.visible = visible;
                }
            """,
        )
        p.x_range.js_on_change("start", callback)
        p.x_range.js_on_change("end", callback)

    # ------------------------------------------------------------------
    # 4. NON-CDS TRACKS — Gantt-chart style
    # ------------------------------------------------------------------
    noncds_types = sorted(noncds_view["type"].unique())
    noncds_y_cursor = (
        cds_track_y_base + 1.8
        if gene_y_positions
        else 1.0
    )

    for feat_type in noncds_types:
        type_df = noncds_view[noncds_view["type"] == feat_type]
        color = FEATURE_COLORS.get(feat_type, "#999999")

        features = type_df.to_dict("records")
        rows = pack_rows(features, gap=200)

        track_label_y = noncds_y_cursor + (len(rows) - 1) * 0.7 / 2
        p.add_layout(
            Label(
                x=view_start, y=track_label_y + 0.5,
                text=feat_type,
                text_font_size="9pt", text_font_style="bold",
                text_color="#555555",
            )
        )

        for ri, row in enumerate(rows):
            y = noncds_y_cursor + ri * 0.7

            lefts   = [f["start"] for f in row]
            rights  = [f["end"] for f in row]
            tops    = [y + 0.28] * len(row)
            bottoms = [y - 0.28] * len(row)
            descs   = [str(f.get("description", "")) for f in row]
            gids    = [str(f.get("gene_id", "")) for f in row]
            strands = [str(f.get("strand", "")) for f in row]
            starts  = [str(f["start"]) for f in row]
            ends    = [str(f["end"]) for f in row]

            src = ColumnDataSource(
                data=dict(
                    left=lefts, right=rights, top=tops, bottom=bottoms,
                    description=descs, gene_id=gids, strand=strands,
                    feat_start=starts, feat_end=ends,
                )
            )
            rend = p.quad(
                left="left", right="right", top="top", bottom="bottom",
                source=src,
                fill_color=color, fill_alpha=0.7,
                line_color="black", line_width=0.6,
            )
            p.add_tools(
                HoverTool(
                    renderers=[rend],
                    tooltips=[
                        ("Type", feat_type),
                        ("Gene", "@gene_id"),
                        ("Description", "@description"),
                        ("Strand", "@strand"),
                        ("Position", "@feat_start \u2013 @feat_end"),
                    ],
                )
            )

            if len(row) <= 40 and view_span < 200_000:
                for f in row:
                    mid = (f["start"] + f["end"]) / 2
                    p.add_layout(
                        Label(
                            x=mid, y=y,
                            text=str(f.get("gene_id", "")),
                            text_font_size="7pt", text_align="center",
                            text_baseline="middle", text_color="white",
                        )
                    )

        noncds_y_cursor += len(rows) * 0.7 + 1.0

    # ------------------------------------------------------------------
    # 5. Y-range — FIXED bounds (no y-zoom / y-pan possible)
    # ------------------------------------------------------------------
    y_bottom = (
        cds_track_y_base - len(cds_packed_rows) * row_height - 0.5
        if cds_packed_rows
        else -1
    )
    y_top = noncds_y_cursor + 0.5
    p.y_range = Range1d(y_bottom, y_top, bounds=(y_bottom, y_top))

    # ------------------------------------------------------------------
    # 6. Strand-direction arrows
    # ------------------------------------------------------------------
    for gene_id, gy in gene_y_positions.items():
        all_exons = chrom_df[
            (chrom_df["type"] == "CDS") & (chrom_df["gene_id"] == gene_id)
        ]
        strand = all_exons.iloc[0]["strand"]
        for _, exon_row in all_exons.iterrows():
            if strand == "+":
                ax = min(int(exon_row["end"]), view_end)
            else:
                ax = max(int(exon_row["start"]), view_start)
            if view_start <= ax <= view_end:
                arrow_dx = view_span * 0.004 * (1 if strand == "+" else -1)
                p.add_layout(
                    Arrow(
                        end=VeeHead(size=6, fill_color="#555555", line_color="#555555"),
                        x_start=ax - arrow_dx, y_start=gy,
                        x_end=ax, y_end=gy,
                        line_color="#555555", line_width=1.5,
                    )
                )

    show(p)
    return p


# ---------------------------------------------------------------------------
# Quick-test scaffold
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Import this module and call genome_browser().  Example:")
    print()
    print('  from genome_browser import genome_browser')
    print('  genome_browser(')
    print('      "Pf3D7_01_v3:29,000-42,000",')
    print('      filtered,')
    print('      "PlasmoDB-54_Pfalciparum3D7_Genome.fasta",')
    print('      "PlasmoDB-68_Pfalciparum3D7_AnnotatedProteins.fasta",')
    print("  )")