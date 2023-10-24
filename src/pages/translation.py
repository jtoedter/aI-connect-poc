import os
import streamlit as st
import base64

from dotenv import load_dotenv
from tinydb import TinyDB
from fpdf import FPDF

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

# Load relevant APIs
load_dotenv()
open_api_key = os.getenv("OPEN_API_KEY")

# Write to app py file
def write():
    with st.spinner("Loading..."):
      st.header("🔤 Translation Agent")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

    # Sidebar to select chatbot model and to clear chat
    st.sidebar.divider()
    st.sidebar.header("Translation Config")
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5 (16K)", "GPT-4 (8K)"))
    clear_button = st.sidebar.button("Clear Chat", key="clear")

    # Sidebar to map model names to OpenAI model IDs
    if model == "GPT-3.5 (16K)":
        model_name = "gpt-3.5-turbo-16k"
    else:
        model_name = "gpt-4"

    # Sidebar to export chat as PDF (not working)
    export_as_pdf = st.sidebar.button("Export Chat (WIP)")

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

    # Initialize chat history database and initialize chat history session
    db = TinyDB("chat_history.json")
    chat_history = db.all()

    # Display chat messages from history on app rerun
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initialise session state variables
    if os.environ.get("OPENAI_API_KEY"):
      if "model_name" not in st.session_state:
          st.session_state["model_name"] = []
      if "generated" not in st.session_state:
          st.session_state["generated"] = []
      if "past" not in st.session_state:
          st.session_state["past"] = []
      if "messages" not in st.session_state:
          st.session_state["messages"] = [{"role": "assistant", "content": "What can I help you with? 何をお手伝いできますか?"}]
      for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

    # Reset session state
    if clear_button:
        st.session_state["model_name"] = []
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["messages"] = [{"role": "assistant", "content": "What can I help you with? 何をお手伝いできますか?"}]

    # Define stream with custom handler for context aware chatbot
    class StreamHandler(BaseCallbackHandler): 
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs):
            self.text += token
            self.container.markdown(self.text)

    # Define chat format in stream
    def display_msg(msg, author):
        """Method to display message on the UI

        Args:
            msg (str): message to display
            author (str): author of the message -user/assistant
        """
        st.session_state["messages"].append({"role": author, "content": msg})
        st.chat_message(author).write(msg)

    # Setup LLM and Agent 
    def setup_agent():
        memory = ConversationBufferMemory()
        llm = ChatOpenAI(temperature=0, model_name=model_name, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        # Define tools - Search & Calculations
        ddg_search = DuckDuckGoSearchRun()
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            ),
            Tool(
                 name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math"
            )]
        # Define system prompts
        system_message = SystemMessage(
           content="""You are expert bi-lingual translator natively fluent in Japanese and English. You only translate between English and Japanese. You never make things up. You never break character. 
           If the user chats to you in English, you will respond in native natural-sounding English. If the user chats to you in Japanese, you will respond in native natural-sounding Japanese. 
           If you are asked to translate between Japanese and English, you will do your best to ensure the translation is accurate, natural-sounding, and contextually-aware. You may ask the user for more information to get more context before translating.          
           
            Complete the objectives above with the following steps:
            1/ When asked to translate between English and Japanese, you will generate increasingly accurate and natural-sounding translations from the input language to the output language
            2/ If the user has not provided any context for the translation, you may ask for more information, such as who the audience will be (example: a guest or client would mean more formal tone; a family or friend would mean more casual tone), or what the format will be (example: email, chat message, phone call, in-person conversation)
            3/ In your first translation, you will ensure you have accurately translated in-verbatim from one language to the other. You will not miss translating any words and you will not make things up.
            4/ For your second translation, you will consider any information and context that you have gathered from the user (example: audience, tone, format) and improve the first translation to make it more relevant and targeted.
            5/ You will then ask yourself: "How can I make the second translation sound more native and natural-sounding in the output language?" You will then do a third translation to the output language ensuring that it is native and natural-sounding.
            6/ You will then compare your third translation to the input language to double-check that you have not made any mistakes or made anything up.
            """

        )
    
        agent_kwargs = {
        "system_message": system_message,
        }

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            verbose=True,
            agent_kwargs=agent_kwargs,
            chain=chain 
            )
        return agent

    agent = setup_agent()

    user_query = st.chat_input(placeholder="Ask me anything & press Enter!")
    if user_query:
        display_msg(user_query, "user")
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            response = agent.run(user_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    write()