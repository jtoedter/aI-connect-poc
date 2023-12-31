import os
import streamlit as st
import requests
import json
import base64

from dotenv import load_dotenv
from fpdf import FPDF
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from langchain.schema import SystemMessage

# Load relevant APIs
load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
open_api_key = os.getenv("OPEN_API_KEY")

# Write to app py file
def write():
    with st.spinner("Loading..."):
      st.header("🌐 Research Agent")

    st.write("⚠️ NOTE - Bugs to be expected & switching bots/agents will reset your conversation.")

    # Sidebar to select chatbot model and to clear chat
    st.sidebar.divider()
    st.sidebar.header("Research Config")
    model = st.sidebar.selectbox("Choose a model:", ("GPT-3.5 Instruct (4K)", "GPT-3.5 (16K)", "GPT-4 (8K)"))
    temp = st.sidebar.selectbox("Select temperature:", ("Low temp", "Mid temp", "High temp"))
    clear_button = st.sidebar.button("Clear Research", key="clear")

    # Sidebar to map model names to OpenAI model IDs
    if model == "GPT-3.5 Instruct (4K)":
        model_name = "gpt-3.5-turbo-instruct"
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
    export_as_pdf = st.sidebar.button("Export Output (WIP)")

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

    # 1. Define tool for web search
    def search(query):
        with st.spinner():
            st.write("Searching the web...")
        
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)

        return response.text

    # 2. Define tool for web scraping
    def scrape_website(objective: str, url: str):
        with st.spinner():
            st.write("Scrabing website...")
        # Scrape website, and also will summarize the content based on objective if the content is too large
        # Objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

        # Define the headers for the request
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }

        # Define the data to be sent in the request
        data = {
            "url": url
        }

        # Convert Python object to JSON string
        data_json = json.dumps(data)

        # Send the POST request
        post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)

        # Check the response status code
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            print("CONTENT:", text)

            if len(text) > 10000:
                output = summary(objective, text)
                
                return output
            else:
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")

    def summary(objective, content):
        with st.spinner():
            st.write("Summarizing search results...")
    
        llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the 'text' below for 'topic': '{objective}' \n
        The text is scraped data from a website so it will have irrelevant information, links and other news, that do not relate to the 'topic'. \n
        Only summarize relevant information and try to keep as much factual information intact: \n
        '{text}' \n\n
        SUMMARY: \n
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"])

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True
        )

        output = summary_chain.run(input_documents=docs, objective=objective)

        return output

    class ScrapeWebsiteInput(BaseModel):
        """Inputs for scrape_website"""
        objective: str = Field(
            description="Objective and topic that users give to the agent")
        url: str = Field(description="URL of the website to be scraped")

    class ScrapeWebsiteTool(BaseTool):
        name = "scrape_website"
        description = "Useful when you need to get data from a website URL, passing both URL and 'topic' to the function. DO NOT make up any URL, the URL should only be from the 'Search' tool results. You will only scrape a website once per user query."
        args_schema: Type[BaseModel] = ScrapeWebsiteInput

        def _run(self, objective: str, url: str):
            return scrape_website(objective, url)

        def _arun(self, url: str):
            raise NotImplementedError("Error here")

    # 3. Create Langchain agent with the tools above
    tools = [
        Tool(
            name="Search",
            func=search,
            description="Useful for when you need to answer questions about the 'topic', current events and data by searching the internet. You should ask targeted questions. You will only search once per user query."
        ),
        ScrapeWebsiteTool(),
    ]

    system_message = SystemMessage(
        content="""You are an expert researcher. You only conduct detailed research on any 'topic' and you only produce facts-based results.
                Your clients are corporate professionals who need recent and relevant facts, information and data.
                You are consistent and you NEVER make things up. You NEVER break character.
                
                Always complete the objective above with the following rules:
                1/ You will only use the 'Search' tool one time when researching and gathering as much recent and relevant information as possible about the 'topic'. 
                2/ You will only 'Scrape Website' tool one time on the most relevant website URL to the 'topic' from the search results.
                3/ Your final research output should be 350 words. You can use bullet points to highlight key information
                4/ You will ALWAYS include reference sources with links at the end of your final output. You will only cite sources and links that you used to write your final output.
                5/ You will not make things up. You should only write based on the facts and data that you have gathered that answers the 'topic'. You will not break character. 
                You will not make things up. You should only write based on the facts and data that you have gathered. You will not break character.
                """
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }

    llm = ChatOpenAI(temperature=temperature, model_name=model_name)
    memory = ConversationSummaryBufferMemory(
        memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

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
    
        if "messages" not in st.session_state:
          st.session_state["messages"] = []

   # Reset session state
    if clear_button:
        st.session_state["messages"] = []

    #4. Use Streamlit to create a web app
    query = st.text_input("What would you like to research? Then press Enter!")
    if query:
        st.write("Doing research on ", query)
        result = agent({"input": query})
        st.info(result["output"])
        st.session_state["messages"].append({"input": query})
        st.session_state["messages"].append({"output": result})

if __name__ == "__main__":
    write()