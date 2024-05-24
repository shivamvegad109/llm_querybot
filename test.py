import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


import os
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

llm = ChatOpenAI()

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# data = chain.invoke({"input": "how can langsmith help with testing?"})
# print(data)

def getLLamaresponse(input_text, no_words):
    # Prompt Template
    template = """
        Write a blog about "{input_text}"
        with approximately {no_words} words.
        """
    prompt = PromptTemplate(input_variables=["input_text", "no_words"], template=template)

    # Generate the response from the LLama2 model
    response = chain.invoke(prompt.format(input_text=input_text, no_words=no_words))
    return response


# Streamlit setup
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📝 Generate Blogs 📝")
st.sidebar.header("🎨 Customization 🎨")

input_text = st.sidebar.text_input("Enter the Blog Topic", "Technology Trends")
no_words = st.sidebar.slider("No of Words", min_value=50, max_value=500, value=100, step=50)

submit = st.sidebar.button("Generate")

st.write("---")

if submit:
    # Check if input fields are not empty
    if not input_text.strip():
        st.warning("Please provide the blog topic.")
    else:
        with st.spinner("🚀 Please wait while generating the blog ..."):
            try:
                blog_content = getLLamaresponse(input_text, no_words)
                if blog_content:
                    st.write(blog_content)
                else:
                    st.error("Failed to generate the blog. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try again later.")
