import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI

# Page setting
st.set_page_config(layout="wide")

# Init langchain
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key="****")# Insert API Key here(**)
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a very helpful assistant"),
     ("user",
      "Based on my Pdf content:{content}. Please answer my question: {question}. Please use the language that I used in the question")]
)
chain = prompt | llm | output_parser

if "content" not in st.session_state:
    st.session_state.content = ""

def main_page():
    st.header("Chat with PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        # Define the temp folder path
        temp_dir = "./temp"
        temp_file = os.path.join(temp_dir, "temp.pdf")

        # Check if the temp directory exists, if not, create it
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Write the uploaded file to the temp file
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Get pdf content
        loader = PyPDFLoader(temp_file)
        pages = loader.load()

        content = ""
        for page in pages:
            content = content + "\n\n" + page.page_content
        st.session_state.content = content

        if st.session_state.content != "":
            col1, col2 = st.columns([4, 6])
            with col1:
                with st.expander("Check PDF Content:", expanded=True):
                    st.write(st.session_state.content)

            with col2:
                question = st.text_input(label="Ask me anything:",
                                         value="Summary the main content ")
                if question != "":
                    with st.spinner("I'm thinking...wait a minute!"):
                        with st.container(border=True):
                            response = chain.invoke({"content": st.session_state.content, "question": question})
                            st.write("Answer:")
                            st.write(response)


if __name__ == '__main__':
    main_page()
