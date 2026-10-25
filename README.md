# 📄 DocuChat: RAG-Powered Document Q&A

Upload a PDF and ask questions. The AI will answer and cite its sources[cite: 2]. This repository contains a Streamlit-based web application that utilizes a Retrieval-Augmented Generation (RAG) pipeline to interact with your documents[cite: 2].

## ✨ Features

* **PDF Uploads**: Seamlessly upload PDF files through the sidebar[cite: 2].
* **Interactive Chat**: A conversational interface that maintains chat history across the session[cite: 2].
* **Source Citations**: Includes an expandable "View Source Context" feature that displays the exact text chunks and page numbers used by the AI[cite: 2].
* **Concise Answers**: The AI is explicitly instructed via a system prompt to keep answers concise and admit when it does not know the answer[cite: 2].

## 🛠 Tech Stack

This project leverages the following Python libraries[cite: 1]:

* **UI Framework**: `streamlit==1.32.0`[cite: 1]
* **LLM Orchestration**: `langchain==0.1.13` and `langchain-community==0.0.29`[cite: 1]
* **Embeddings & Models**: `langchain-openai==0.1.1`[cite: 1]
* **Vector Database**: `chromadb==0.4.24`[cite: 1]
* **Document Parsing**: `pypdf==4.1.0`[cite: 1]
* **Environment Management**: `python-dotenv==1.0.1`[cite: 1]

## 🚀 Installation & Setup

### 1. Prerequisites
You will need an active OpenAI API key, as the application relies on `OpenAIEmbeddings` and the `ChatOpenAI` model[cite: 2].

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

### 3. Install Dependencies
Install the required packages listed in the `requirements.txt` file[cite: 1]:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
The application uses `load_dotenv()` to read environment variables[cite: 2]. Create a `.env` file in the root directory and add your API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 💻 Usage

1. **Start the Application**: 
   Run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
2. **Upload a Document**: Open the app in your browser, navigate to the sidebar labeled "1. Upload Document", and upload a PDF[cite: 2].
3. **Wait for Processing**: The app will save the file temporarily, vectorize it, and display a success message when ready[cite: 2].
4. **Chat**: Use the input box to ask questions about your document[cite: 2].
5. **Verify Sources**: Click the "📚 View Source Context (Citations)" expander under the assistant's reply to see the exact context used to generate the response[cite: 2].

## 🧠 Architecture (Under the Hood)

The app utilizes a classic RAG (Retrieval-Augmented Generation) pipeline defined in `app.py`[cite: 2]:

* **Document Loading**: Uploaded PDFs are written to a temporary file and parsed using `PyPDFLoader`[cite: 2].
* **Text Chunking**: Text is split using `RecursiveCharacterTextSplitter` into chunks of 1000 characters with a 200-character overlap to prevent context loss at boundaries[cite: 2].
* **Vector Storage**: `OpenAIEmbeddings` convert the text chunks into vectors, which are then stored locally in memory using `Chroma`[cite: 2].
* **Retrieval & Generation**: 
    * The vector store acts as a retriever, fetching the top 3 (`k=3`) most relevant chunks for the user's query[cite: 2].
    * These chunks are passed to the `gpt-3.5-turbo` model, which is set to a `temperature` of `0` for predictable outputs[cite: 2]. 
    * LangChain's `create_stuff_documents_chain` and `create_retrieval_chain` connect the retrieved context to the LLM to generate the final answer[cite: 2].
