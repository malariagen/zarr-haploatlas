import streamlit as st

from src.utils import build_chunk_index, load_reference_files, parse_loci_from_input

st.set_page_config(layout="wide")

chunk_index_df  = build_chunk_index()
reference_files = load_reference_files()

RAW_USER_INPUT = st.text_input(
    "Enter a genomic region",
    value = "PF3D7_0709000[72-76,326] Pf3D7_04_v3[104205,139150-139156]",
    help = "For nucleotide haplotypes, use the format: `chromosome[start-end]` or `chromosome[position1,position2]`. \
        For amino acid haplotypes, use the format: `protein[start-end]` or `protein[position1,position2]`. \
        Multiple loci separated by spaces. Use PlasmoDB naming conventions (i.e., `Pf3D7_??_v3` for loci and `PF3D7_???????` for proteins)"
)

loci = parse_loci_from_input(RAW_USER_INPUT)



st.dataframe(loci)