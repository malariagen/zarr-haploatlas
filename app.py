import streamlit as st
from tabs.home     import render as render_home
from tabs.order    import render as render_order
from tabs.checkout import render as render_checkout

st.set_page_config(layout="wide", page_title="Variant Marketplace", page_icon="assets/logo.svg")
st.logo("assets/logo.svg", size="large")

if not st.user.is_logged_in:
    st.title("Variant Marketplace")
    st.button("Sign in with Google", on_click=st.login, args=["google"])
    st.stop()

with st.sidebar:
    st.write(f"Signed in as **{st.user.email}**")
    st.button("Sign out", on_click=st.logout)

tab_home, tab_order, tab_checkout = st.tabs(["Home", "Order", "Checkout"])

with tab_home:
    render_home()

with tab_order:
    render_order()

with tab_checkout:
    render_checkout()
