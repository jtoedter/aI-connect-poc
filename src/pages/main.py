import streamlit as st

# Write to app py file
def write():
    with st.spinner("Loading..."):
        st.header("🤖 Welcome to Litmus AI Connect! (**ALPHA**)")

    st.write("""
    To use a bot or an agent, simply click on them on the navigation pane. Enjoy!
      
    ⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.
    """)
            
    st.success("""
    Available Litmus AI Connect Agents in GPT 3.5 or GPT 4:
    - Chatbot: Chat away with our helpful assistant with general queries. Note, our chatbot cannot access the internet & perform complex calculations (yet)!
    - Research Agent: Our research assistant will search the internet based on your queries & write up a short summary with links to references!
      """)

    st.warning("""
    Under Development Litmus AI Connect Agents:
    - Translation Agent: <In Development>
    """)

if __name__ == "__main__":
    write()