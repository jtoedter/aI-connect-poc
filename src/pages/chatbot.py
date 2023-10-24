import os
import streamlit as st
import base64

from dotenv import load_dotenv
from streamlit_chat import message
from tinydb import TinyDB
from fpdf import FPDF

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()
open_api_key = os.getenv("OPEN_API_KEY")

# Write to main app py file
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading..."):
      st.header("💬 Chatbot")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

    # Consolidated from streaming py file
    class StreamHandler(BaseCallbackHandler): 
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text)

    # Context aware chatbot code ##
   
    # Initialize chat history
    db = TinyDB("chat_history.json")

    # Define chat format
    def display_msg(msg, author):
        """Method to display message on the UI

        Args:
            msg (str): message to display
            author (str): author of the message -user/assistant
        """
        st.session_state["messages"].append({"role": author, "content": msg})
        st.chat_message(author).write(msg)

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.divider()
    st.sidebar.header("Chatbot Config")
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5 (16K)", "GPT-4 (8K)"))
    clear_button = st.sidebar.button("Clear Chat", key="clear")

    # Map model names to OpenAI model IDs
    if model == "GPT-3.5 (16K)":
        model_name = "gpt-3.5-turbo-16k"
    else:
        model_name = "gpt-4"

    # Export as PDF
    export_as_pdf = st.sidebar.button("Export Chat")

    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download PDF</a>'

    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(40, 10, "Not working yet! Sorry!")
        
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "PDF")

        st.sidebar.markdown(html, unsafe_allow_html=True)

    # Initialise chat history session state variables
    chat_history = db.all()

    # Display chat messages from history on app rerun
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialise session state variables
    if os.environ.get("OPENAI_API_KEY"):
      if "model_name" not in st.session_state:
          st.session_state[model_name] = []
      if "generated" not in st.session_state:
          st.session_state["generated"] = []
      if "past" not in st.session_state:
          st.session_state["past"] = []
      if "messages" not in st.session_state:
          st.session_state["messages"] = [
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "assistant", "content": "How can I help you today?"}
          ]
      for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

    # reset everything
    if clear_button:
        st.session_state[model_name] = []
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."},
             {"role": "assistant", "content": "What can I help you with next?"}
        ]

    @st.cache_resource
    def setup_chain():
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)
        return chain

    chain = setup_chain()
    user_query = st.chat_input("Ask me anything & press Enter!")
    if user_query:
        display_msg(user_query, "user")
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            response = chain.run(user_query, callbacks=[st_cb])
            st.session_state['messages'].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    write()