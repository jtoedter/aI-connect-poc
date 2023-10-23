"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading..."):
        st.header("🤖 Welcome to Litmus AI Connect!")

    st.write("""
    To use a bot or an agent, simply click on them on the navigation pane. Enjoy!
      
    ⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.
    """)
            
    st.success("""
    Available Litmus AI Connect Agents:
    - Chatbot: Choose between GPT 3.5 or GPT 4 & chat away!
    - Research Agent: <In Development>
      """)

    st.warning("""
    Under Development Litmus AI Connect Agents:
    - Translation Agent: <In Development>
    """)
