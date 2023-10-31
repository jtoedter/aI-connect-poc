"""Main module for the streamlit app"""
import streamlit as st

import awesome_streamlit as ast
import src.pages.main
import src.pages.chatbot
import src.pages.research
import src.pages.translation
import src.pages.translation_new

PAGES = {
    "🤖 Main": src.pages.main,
    "💬 Chatbot": src.pages.chatbot,
    "🌐 Research Agent": src.pages.research,
    "🔤 Translation Agent <WIP>": src.pages.translation,
    "🔤 Translation Test <TEST>": src.pages.translation_new,
}

def main():
    """Main function of the App"""
    st.set_page_config(page_title="Litmus AI Connect POC", page_icon="🐉")

    st.sidebar.title("🐉 Litmus AI Connect POC (**ALPHA**)")
    st.sidebar.header("Navigation")
    selection = st.sidebar.radio("Go to:", list(PAGES.keys()))

    page = PAGES[selection]
    with st.spinner(f"Loading {selection}..."):
        ast.shared.components.write_page(page)

    st.sidebar.divider()

    st.sidebar.info("""
    **Litmus Japan**
    
    [Website](https://www.litmus-jp.com/) - 
    [LinkedIn](https://www.linkedin.com/company/litmus-jp/)               
    """)

    st.sidebar.markdown(" © 2023 Litmus K.K. All Rights Reserved.")

if __name__ == "__main__":
    main()