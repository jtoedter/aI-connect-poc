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
from langchain import PromptTemplate

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
    i_language = st.sidebar.selectbox("Your input language:", ("English", "Japanese"))
    o_language = st.sidebar.selectbox("Your output language:", ("Japanese", "English", "Surprise me!"))
    model = st.sidebar.selectbox("Choose a model:", ("GPT-3.5 (4K)", "GPT-3.5 (16K)", "GPT-4 (8K)"))
    temp = st.sidebar.selectbox("Select temperature:", ("Low temp", "Mid temp", "High temp"))
    clear_button = st.sidebar.button("Clear Translation", key="clear")

    # Sidebar to map input language to system prompts
    if i_language == "English":
        input_language = "English"
    else:
        input_language = "Japanese"

    # Sidebar to map output language to system prompts
    if o_language == "English":
        output_language = "English"
    if o_language == "Japanese":
        output_language = "Japanese"
    else:
        output_language = "Surprise me!"

    # Sidebar to map model names to OpenAI model IDs
    if model == "GPT-3.5 (4K)":
        model_name = "gpt-3.5-turbo"
    if model == "GPT-4 (16K)":
        model_name = "gpt-3.5-turbo-16k"
    else:
        model_name = "gpt-4"

    # Sidebar to map temperatue to OpenAI model IDs
    if temp == "Low temp":
        temperature = "0"
    if temp == "Mid temp":
        temperature = "0.3"
    else:
        temperature = "0.7"

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
      
        # To clear chat history after swtching chatbot
        current_page = __name__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

            # Session initiation if blank
        if "model_name" not in st.session_state:
            st.session_state["model_name"] = []
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "How can I help you today?"}
            ]
        for msg in st.session_state["messages"]:
                    st.chat_message(msg["role"]).write(msg["content"])

    # Reset session state
    if clear_button:
        st.session_state["model_name"] = []
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["messages"] = [
            {"role": "assistant", "content": "What can I help you with next?"}
        ]

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
        llm = ChatOpenAI(temperature=temperature, model_name=model_name, streaming=True)
        chain = ConversationChain(llm=llm, memory=memory, verbose=True)

        # Define tools - Search & Calculations
        ddg_search = DuckDuckGoSearchRun()
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions"
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="Useful for when you need to answer questions involving mathematical calculations"
            )]

        map_prompt = {
        "input_language": input_language,
        "output_language": output_language,
        }

        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["input_language", "output_language"])

        # Define system prompts
        system_message = SystemMessage(
           content="""You are expert bi-lingual translator in {input_language} and {output_language}. You are a native speaker and professionally fluent in both of these languages.
            Your output is only in the {output_language} which is a translation from the user input which you received in {input_language}.
            Your output should always be accurately translated and naturally-sounding in {output_language}.
            You never make things up. You never break character. 
            """
        )
    
        agent_kwargs = {
        "system_message": system_message,
        }

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            chain=chain,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            verbose=True,
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            agent_kwargs=agent_kwargs,
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