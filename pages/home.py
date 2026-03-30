import streamlit as st

st.image("assets/logo.svg", width=120)
st.title("Variant Marketplace", anchor=False)
st.markdown(
    """
    A tool for querying, building, and exporting haplotype data from *P. falciparum* variant calls.

    **How it works:**

    1. **Order** — Enter one or more genomic loci (amino acid or nucleotide coordinates).
       The app looks up variant metadata and lets you preview the sites of interest.
       Select a formatting strategy for heterozygous calls, then build — one TSV file
       is saved per locus.

    2. **Checkout** — Browse all saved haplotype files, select any combination,
       and merge them into a single TSV joined on sample ID. Download the result.
    """
)
