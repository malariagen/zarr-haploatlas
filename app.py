import streamlit as st

st.set_page_config(layout="wide", page_title="Variant Marketplace", page_icon="assets/logo.svg")
st.logo("assets/logo.svg", size="large")

order_page   = st.Page("pages/order.py",   title="Order",   icon=":material/shopping_cart:")
checkout_page = st.Page("pages/inspect.py", title="Checkout", icon=":material/dataset_linked:")

pg = st.navigation([order_page, checkout_page])
pg.run()
