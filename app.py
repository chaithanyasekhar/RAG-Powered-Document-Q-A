import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- 1. UI Configuration ---
st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")
st.title("📄 DocuChat: RAG-Powered Document Q&A")
st.markdown("Upload a PDF and ask questions. The AI will answer and cite its sources.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- 2. Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Step A: Load the PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Step B: Chunk the text
            # We split the text into 1000-character chunks with a 200-character overlap 
            # so context isn't lost at the boundaries.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Step C: Create Embeddings and Store in ChromaDB
            embeddings = OpenAIEmbeddings()
            vector_store = Chroma.from_documents(chunks, embeddings)
            
            # Save to session state so it persists across user questions
            st.session_state.vector_store = vector_store
            
            # Cleanup temp file
            os.unlink(tmp_file_path)
            st.success("Document processed and vectorized successfully!")

# --- 3. Main Chat Interface ---
# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask a question about your document..."):
    if st.session_state.vector_store is None:
        st.error("Please upload a document first.")
    else:
        # Add user question to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # --- 4. RAG Retrieval & Generation ---
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Setup the LLM
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                
                # Setup the System Prompt
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Keep the answer concise.\n\n"
                    "{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # Create the chains
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 3} # Retrieve the 3 most relevant chunks
                )
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Execute the chain
                response = rag_chain.invoke({"input": user_query})
                
                answer = response["answer"]
                source_documents = response["context"]
                
                # Display Answer
                st.markdown(answer)
                
                # The "Shortlist" Feature: Displaying Citations
                with st.expander("📚 View Source Context (Citations)"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                        st.text(doc.page_content)
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})