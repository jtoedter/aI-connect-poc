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

# Initialize debounce variables
last_call_time = 0
debounce_interval = 2  # Set the debounce interval (in seconds) to your desired value

def debounce_replicate_run(llm, prompt, max_len, temperature, top_p, API_TOKEN):
    global last_call_time
    print("last call time: ", last_call_time)

    # Get the current time
    current_time = time.time()

    # Calculate the time elapsed since the last call
    elapsed_time = current_time - last_call_time

    # Check if the elapsed time is less than the debounce interval
    if elapsed_time < debounce_interval:
        print("Debouncing")
        return "Hello! You are sending requests too fast. Please wait a few seconds before sending another request."


    # Update the last call time to the current time
    last_call_time = time.time()
    
    output = replicate.run(llm, input={"prompt": prompt + "Assistant: ", "max_length": max_len, "temperature": temperature, "top_p": top_p, "repetition_penalty": 1}, api_token=API_TOKEN)
    return output

# Load relevant APIs
load_dotenv()
open_api_key = os.getenv("OPEN_API_KEY")

# Write to app py file
st.spinner("Loading...")
st.header("🔤 Translation Agent")
st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

# Constants
PRE_PROMPT = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as Assistant."

MODELS = {
    "GPT-3.5 (4K)": "gpt-3.5-turbo",
    "GPT-3.5 (16K)": "gpt-3.5-turbo-16k",
    "GPT-4 (8K)": "gpt-4",
}

# Session State Variables
DEFAULT_TEMP = 0.0
DEFAULT_MODEL = "gpt-3.5-turbo-16k"
DEFAULT_PRE_PROMPT = PRE_PROMPT

# Render sidebar config additions to main sidebar
st.sidebar.divider()
st.sidebar.header("Translation Config")

# Initialize sessions
def setup_session_state():
    if os.environ.get("OPENAI_API_KEY"):

        #Set default session
        st.session_state.setdefault("chat_history", [])
        model_name = st.sidebar.selectbox("Choose a model:", list(MODELS.keys()), key="model_name")
        st.session_state.setdefault("model_name", DEFAULT_MODEL)
        st.session_state.setdefault("temperature", DEFAULT_TEMP)
        st.session_state.setdefault("pre_prompt", DEFAULT_PRE_PROMPT)
        st.session_state.setdefault["messages"] = [
            {"role": "assistant", "content": "How can I help you today?"}
        ]
    for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

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

def render_sidebar():
    temperature = st.session_state["temperature"] = st.sidebar.slider("Temperature:", min_value=0.0, max_value=1.0, value=DEFAULT_TEMP, step=0.1, key="temperature")
    new_prompt = st.sidebar.text_area("Additional prompt before the chat starts. Add here if required:", DEFAULT_PRE_PROMPT, height=50)
    if new_prompt != DEFAULT_PRE_PROMPT and new_prompt != "" and new_prompt is not None:
        st.session_state["pre_prompt"] = DEFAULT_PRE_PROMPT + new_prompt + "\n\n"
    else:
        st.session_state["pre_prompt"] = DEFAULT_PRE_PROMPT

    clear_button = st.sidebar.button("Clear Translation", key="clear")
    if clear_button:
        st.session_state["model_name"] = []
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["messages"] = [
            {"role": "assistant", "content": "What can I help you with next?"}
    ]

# Initialize chat history database and initialize chat history session
def render_chat_history():
    response_container = st.container()
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_msg(msg, author):
    message_placeholder = st.empty()
    full_response = ""
    string_dialogue = st.session_state["pre_prompt"]

    for dict_message in st.session_state.chat_history:
        speaker = "user" if dict_message["role"] == "user" else "assistant"
        string_dialogue += f"{speaker}: {dict_message['content']}\n\n"
    
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state["messages"].append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
     
    message_placeholder.markdown(full_response)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# Define stream with custom handler for context aware chatbot
class StreamHandler(BaseCallbackHandler): 
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

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

    #map_prompt = {
    #"input_language": input_language,
    #"output_language": output_language,
    #}

    #map_prompt_template = PromptTemplate(
    #    template=map_prompt, input_variables=["input_language", "output_language"])

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
    #    map_prompt=map_prompt_template,
    #    combine_prompt=map_prompt_template,
        agent_kwargs=agent_kwargs,
        )
    return agent

agent = setup_agent()

def handle_user_input():
    user_input = st.chat_input(placeholder="Ask me anything & press Enter!")
    if user_input:
        display_msg(user_input, "user")
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            response = agent.run(user_input, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})

def render_app():
    setup_session_state()
    render_sidebar()
    render_chat_history()
    display_msg()
    setup_agent()
    handle_user_input()
    
def write():
    render_app()

if __name__ == "__main__":
    write()