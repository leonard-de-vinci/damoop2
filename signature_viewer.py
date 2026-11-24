"""Standalone Signature Viewer (same page as in app.py). Run: streamlit run signature_viewer.py"""
import streamlit as st

from views.signature_viewer import render

st.set_page_config(page_title="Signature viewer", layout="wide")
st.title("Signature viewer")
render()
