import streamlit as st
from langchain_community.llms import Ollama
from langchain_openai import OpenAIEmbeddings
from vector_file import *
import time
# Load the LLama2 model outside the function
llm = Ollama(model="llama2")
embedding = OpenAIEmbeddings()

# Streamlit setup
st.set_page_config(
    page_title="Blog Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_blog(input_text, no_words):
    if not input_text.strip():
        st.warning("Please provide the blog topic.")
        return

    with st.spinner("🚀 Please wait while generating the blog ..."):
        try:
            vector_data = visualize_data(embedding)
            if vector_data["Page Content"].astype(str).str.contains(input_text, case=False).any():
                results = load_model_to_local(embedding, input_text)
                refrence = results.get('result')
                print(refrence,"\n\n")
                blog_content_with_vector_prompt = getLLamaresponse_with_vector_prompt(input_text, no_words, refrence)
                print(blog_content_with_vector_prompt)
                if blog_content_with_vector_prompt:
                    with open("read_file.txt", "a") as f:
                        f.write(blog_content_with_vector_prompt+"\n")
                    doc_save_local("read_file.txt",embedding)
                    st.write(blog_content_with_vector_prompt)
                else:
                    st.error("Failed to generate the blog. Please try again later.")
                # results = load_model_to_local(embedding, input_text)
                # if results:
                #     st.write(f"**QUERY:**\n\n {results.get('query')}\n\n",
                #              f"**RESULT:**\n\n {results.get('result')}")                
                # else:
                #     st.error("Failed to generate query results. Please try again later.")
            else:
                print("Not Found in local vector")
                blog_content = getLLamaresponse(input_text, no_words)
                if blog_content:
                    with open("read_file.txt", "a") as f:
                        f.write(blog_content+"\n")
                    doc_save_local("read_file.txt",embedding)
                    st.write(blog_content)
                else:
                    st.error("Failed to generate the blog. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}. Please try again later.")

def main():
    st.title("📝 Generate Blogs 📝")

    st.sidebar.header("🎨 Customization 🎨")
    input_text = st.sidebar.text_input("Enter the Blog Topic", "Latest Innovations")
    no_words = st.sidebar.slider("No of Words", min_value=50, max_value=500, value=100, step=50)
    submit = st.sidebar.button("Generate")

    st.sidebar.header("❓For Query❓")
    input_query = st.sidebar.text_input("Enter your query")
    submit_query = st.sidebar.button("Get query")

    get_all_vectors = st.sidebar.button("Get All Vectors")

    delete_documents = st.sidebar.button("Delete Document")
    st.write("---")
    
    if submit:
        generate_blog(input_text, no_words)

    if get_all_vectors:
        st.write(visualize_data(embedding))

    if submit_query:
        results = load_model_to_local(embedding, input_query)
        if results:
            st.write(f"**QUERY:**\n\n {results.get('query')}\n\n",
                        f"**RESULT:**\n\n {results.get('result')}")                
        else:
            st.error("Failed to generate query results. Please try again later.")

    if delete_documents:
        doc = "read_file.txt"
        delete_document(doc,embedding)
        with open(doc, 'r+') as file:
            file.truncate(0)  
        st.write("Document Deleted Successfully")
    st.components.v1.html(
        """
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 10000;">
            <script>
                window.embeddedChatbotConfig = {
                    chatbotId: "F_a-f9Ik3Bkr6bSa1rxZM",
                    domain: "www.chatbase.co"
                }
            </script>
            <script
                src="https://www.chatbase.co/embed.min.js"
                chatbotId="F_a-f9Ik3Bkr6bSa1rxZM"
                domain="www.chatbase.co"
                defer>
            </script>
        </div>
        <style>
            div[style*='position: fixed; bottom: 10px; right: 10px;'] {
                width: 350px;
                height: 500px;
                z-index: 10000;  /* Ensure the chatbot stays on top */
            }
            @media (max-width: 768px) {
                div[style*='position: fixed; bottom: 10px; right: 10px;'] {
                    width: 300px;
                    height: 400px;
                }
            }
            @media (max-width: 480px) {
                div[style*='position: fixed; bottom: 10px; right: 10px;'] {
                    width: 250px;
                    height: 350px;
                }
            }
        </style>
        """,
        height=600,  # Adjust height to fit the content
        scrolling=True,
    )


if __name__ == "__main__":
    main()
