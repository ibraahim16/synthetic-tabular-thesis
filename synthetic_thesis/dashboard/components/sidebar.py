import streamlit as st
from config import PAGES

def show_sidebar():
    st.sidebar.title("Menü")

    selected_page = st.sidebar.radio(
        "",
        PAGES
    )

    return selected_page