"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading..."):
      st.header("🔤 Translation Agent")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")