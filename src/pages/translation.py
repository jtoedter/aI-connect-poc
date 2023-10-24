import streamlit as st

def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading..."):
      st.header("🔤 Translation Agent")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

if __name__ == "__main__":
    write()