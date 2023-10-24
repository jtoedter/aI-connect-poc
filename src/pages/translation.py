import streamlit as st

# Write to app py file
def write():
    with st.spinner("Loading..."):
      st.header("🔤 Translation Agent")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

if __name__ == "__main__":
    write()