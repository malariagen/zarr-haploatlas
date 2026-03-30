import streamlit as st

st.set_page_config(layout="wide", page_title="Variant Marketplace", page_icon="assets/logo.svg")
st.logo("assets/logo.svg", size="large")

home_page     = st.Page("pages/home.py",     title="Home",     icon=":material/home:")
order_page    = st.Page("pages/order.py",    title="Order",    icon=":material/shopping_cart:")
checkout_page = st.Page("pages/checkout.py", title="Checkout", icon=":material/dataset_linked:")

pg = st.navigation([home_page, order_page, checkout_page])
pg.run()
