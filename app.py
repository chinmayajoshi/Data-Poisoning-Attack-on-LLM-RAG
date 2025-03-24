import streamlit as st
import json
import numpy as np
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# Disable Streamlit's automatic PyTorch detection which might be causing the error
os.environ["STREAMLIT_TORCH_DETECTION"] = "0"

# Set page config
st.set_page_config(page_title="RAG App with Groq", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load FAISS index and chunks
@st.cache_resource
def load_resources():
    try:
        index = faiss.read_index("faiss_store/faiss_index.bin")
        with open("faiss_store/chunks.json", "r") as f:
            chunks = json.load(f)
        model_name = "intfloat/multilingual-e5-small"
        embed_model = HuggingFaceEmbeddings(model_name=model_name, cache_folder="cached_models/")
        return index, chunks, embed_model
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

# Initialize Groq LLM
@st.cache_resource
def init_llm():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            api_key = st.secrets.get("GROQ_API_KEY", "")
            if not api_key:
                st.warning("GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
                return None, "No model loaded"
        
        # Set model name here
        model_name = "llama3-70b-8192"
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.1
        )
        return llm, model_name
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None, "Error loading model"

# App UI
st.title("RAG Search App")

# Try to load resources with error handling
try:
    index, chunks, embed_model = load_resources()
    llm, model_name = init_llm()

    # Define the RAG prompt template
    rag_prompt = PromptTemplate(
        input_variables=["query", "context", "chat_history"],
        template="""
        You are a helpful assistant that answers questions based on the provided context.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {query}
        
        Answer the question based on the context provided and previous conversation if relevant. 
        If the answer is not contained within the context, say "I don't have enough information to answer this question" and suggest a better question to ask.
        """
    )

    # Create RAG chain
    if llm:
        rag_chain = LLMChain(llm=llm, prompt=rag_prompt)
    else:
        st.error("Cannot create RAG chain without LLM.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=20, value=5)
        
        # Model information display
        st.subheader("Model Information")
        st.info(f"LLM: Groq - {model_name}\nEmbedding: intfloat/multilingual-e5-small")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            try:
                st.rerun()
            except Exception:
                st.success("Chat history cleared. Please refresh the page.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    query = st.chat_input("Ask a question about your documents...")

    if query and index and chunks and embed_model and llm:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    # Format chat history for context
                    chat_history = ""
                    for msg in st.session_state.messages[:-1]:  # Exclude the current query
                        chat_history += f"{msg['role']}: {msg['content']}\n"
                    
                    # Get query embedding
                    query_embedding = embed_model.embed_query(query)
                    
                    # Search in FAISS
                    distances, indices = index.search(np.array([query_embedding]), k=top_k)
                    
                    # Retrieve chunks
                    retrieved_indices = indices[0]
                    retrieved_chunks = [chunks[int(idx)] for idx in retrieved_indices]
                    
                    # Combine chunks into context
                    context = "\n\n".join(retrieved_chunks)
                    
                    # Generate response with RAG
                    response = rag_chain.invoke({"query": query, "context": context, "chat_history": chat_history})
                    
                    # Display response
                    response_text = response.get("text", "No response generated")
                    st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Show retrieved documents in expander
                with st.expander("Retrieved Documents"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Document {i+1}** (Score: {1 - distances[0][i]:.4f})")
                        st.text(chunk)
                        
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
                
except Exception as e:
    st.error(f"Critical application error: {str(e)}")
    st.info("Try running the app with '--server.enableStaticServing false' flag or restart the application.")