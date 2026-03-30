import streamlit as st
from tabs.home     import render as render_home
from tabs.order    import render as render_order
from tabs.checkout import render as render_checkout

st.set_page_config(layout="wide", page_title="Variant Marketplace", page_icon="assets/logo.svg")
st.logo("assets/logo.svg", size="large")

tab_home, tab_order, tab_checkout = st.tabs(["Home", "Order", "Checkout"])

with tab_home:
    render_home()

with tab_order:
    render_order()

with tab_checkout:
    render_checkout()
