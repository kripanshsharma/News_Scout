import os
import streamlit as st
import pickle
import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up the Streamlit page configuration with an improved theme
st.set_page_config(
    page_title="NewScout: Chat with Your Articles 📰",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    h1 {
        color: #00CCFF;
    }
    h2 {
        color: #FFFFFF;
    }
    .block-container {
        padding: 2rem;
    }
    .sidebar-content {
        background-color: #2C2C2C;
    }
    .sidebar .sidebar-content {
        padding-top: 20px;
        padding-left: 15px;
    }
    input[type="text"] {
        border-radius: 5px;
        padding: 8px;
        border: 1px solid #ddd;
        font-size: 16px;
    }
    .css-1n76uvr.e1fqkh3o7 {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 NewScout: Chat with Your Articles 🌐")
st.sidebar.title("Enter Your Article URLs 📎")


st.sidebar.markdown("### Enter up to 3 article URLs to analyze 📰🌐:")
urls = [st.sidebar.text_input(f"URL {i+1}", placeholder="Enter URL here...") for i in range(3)]

bot_name = st.sidebar.text_input("Name your bot:", value="NewScout")

# Function to fetch article content from a single link
def fetch_article_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Failed to fetch article at {url}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No Title'
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        
        return {"title": title, "content": content}
    except Exception as e:
        st.error(f"Error fetching article from {url}: {e}")
        return None

# Process content into embeddings and vector store
def process_content_for_bot(article_content):
    documents = [Document(page_content=article_content)]
    
    # Create HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Generate and store embeddings in FAISS
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

# Set up the conversational bot using Ollama (LLaMA 2)
def setup_chain(vector_store):
    llm = Ollama(model="llama2")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )
    return chain

if 'history' not in st.session_state:
    st.session_state['history'] = {}

if 'processed_articles' not in st.session_state:
    st.session_state['processed_articles'] = []


if st.sidebar.button("Process URLs"):
    if any(urls):
        with st.spinner("Loading articles from URLs... 🌐📥"):
            articles = []
            for url in urls:
                if url.strip():
                    article_details = fetch_article_content(url)
                    if article_details:
                        articles.append(article_details)
            
        with st.spinner("Splitting text into manageable chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = [Document(page_content=article['content']) for article in articles]
            chunks = text_splitter.split_documents(docs)

        with st.spinner("Generating embeddings..."):
            vector_store = FAISS.from_documents(chunks, HuggingFaceEmbeddings())

        st.sidebar.success("Articles processed successfully! ✅")

        # Store articles and vector store in session state
        st.session_state['processed_articles'] = articles
        st.session_state['vector_store'] = vector_store

        # Display article summaries
        st.header("📰 Article Summaries")
        for i, article in enumerate(articles):
            st.subheader(f"Article {i + 1}")
            st.write(f"**Title**: {article['title']}")
            st.write(article['content'][:500] + "...")
            st.write("---")
    else:
        st.sidebar.error("Please enter at least one URL to proceed.")

# Allow user to select an article for chat
if 'processed_articles' in st.session_state:
    selected_article = st.selectbox(
        "Select an article to chat with:",
        options=[article['title'] for article in st.session_state['processed_articles']]
    )

# Input field for user query
if selected_article:
    query = st.text_input(f"Ask a question about the selected article '{selected_article}': ❓")
    
    if query:
        # Initialize chat history if not already present
        if selected_article not in st.session_state['history']:
            st.session_state['history'][selected_article] = []

        # Retrieve the vector store and set up the bot
        if 'vector_store' in st.session_state:
            vector_store = st.session_state['vector_store']
            chain = setup_chain(vector_store)
            chat_history = st.session_state['history'][selected_article]

            # Get response
            result = chain({"question": query, "chat_history": chat_history}, return_only_outputs=True)

            st.header(f"Answer from {bot_name}")
            st.write(result["answer"])

            # Update chat history
            st.session_state['history'][selected_article].append((query, result["answer"]))

            # Show conversation history
            if st.checkbox("Show previous conversations"):
                st.header(f"Conversation History with '{selected_article}'")
                for idx, entry in enumerate(st.session_state['history'][selected_article]):
                    st.subheader(f"Query {idx + 1}")
                    st.write(f"**Question:** {entry[0]}")
                    st.write(f"**Answer:** {entry[1]}")