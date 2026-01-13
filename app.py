import streamlit as st
import os
import json
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile


def resolve_pdf_path(pdf_path_candidate: str):
    """Try several ways to resolve a user-provided PDF path to an existing file."""
    if not pdf_path_candidate:
        return None

    candidates = []
    # raw input
    candidates.append(pdf_path_candidate)
    # expanduser
    candidates.append(os.path.expanduser(pdf_path_candidate))
    # relative to current working dir
    candidates.append(os.path.join(os.getcwd(), pdf_path_candidate))
    # relative to the script directory (Project2)
    candidates.append(os.path.join(os.path.dirname(__file__), pdf_path_candidate))
    # common subfolder (Project2)
    candidates.append(os.path.join(os.path.dirname(__file__), os.path.basename(pdf_path_candidate)))

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            continue
    return None

# Page configuration
st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-citation {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# System prompt
QNA_SYSTEM_MESSAGE = """
You are an AI assistant designed to support healthcare professionals and patients in efficiently
accessing evidence-based medical knowledge. Your task is to provide concise, accurate, and clinically
relevant answers based on the context provided from trusted medical references such as the Merck Manuals.

User input will include the necessary context for you to answer their questions. This context
will begin with the token: ###Context

When crafting your response:
- Use only the provided context to answer the question.
- If the answer is found in the context, respond with clear, guideline-driven summaries.
- Include the section title and page number (or subsection reference) as the source.
- If the question is unrelated to the context or the context is empty, clearly respond with:
  "This information is not available in the provided context. For safe and personalized medical
   advice, please consult a licensed healthcare professional."

Format your response as:
Answer: [Your answer based on context]

Source: [Section title, page number or subsection reference]
"""

QNA_USER_MESSAGE_TEMPLATE = """
###Context
{context}

###Question
{question}

Instructions:
Use the context above to answer the question as accurately as possible.
If the context contains the answer, provide a concise, evidence-based summary and cite the source.
If the context is empty or unrelated, respond with the standard safety message.
"""


def load_api_credentials():
    """Load API credentials from config or environment"""
    try:
        # Try several possible config locations (cwd and script directory)
        possible_paths = [
            os.path.join(os.getcwd(), 'config.json'),
            os.path.join(os.path.dirname(__file__), 'config.json'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, 'r') as file:
                    config = json.load(file)
                    return config.get("OPENAI_API_KEY"), config.get("OPENAI_API_BASE")

        # Fall back to Streamlit secrets or environment variables
        api_key = None
        api_base = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY") if isinstance(st.secrets, dict) else st.secrets.get("OPENAI_API_KEY")
            api_base = st.secrets.get("OPENAI_API_BASE") if isinstance(st.secrets, dict) else st.secrets.get("OPENAI_API_BASE")
        except Exception:
            # st.secrets may be empty or not behave like a dict; ignore errors
            pass

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_base:
            api_base = os.getenv("OPENAI_API_BASE")

        return api_key, api_base
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return None, None


def initialize_openai_client(api_key, api_base=None):
    """Initialize OpenAI client"""
    if api_base:
        return OpenAI(api_key=api_key, base_url=api_base)
    return OpenAI(api_key=api_key)


def process_pdf(pdf_file, api_key, api_base):
    """Process uploaded PDF and create vector store"""
    with st.spinner("Processing PDF... This may take a few minutes."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyMuPDFLoader(tmp_path)
            documents = loader.load()
            st.success(f"✅ Loaded {len(documents)} pages from PDF")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name='cl100k_base',
                chunk_size=512,
            )
            document_chunks = text_splitter.split_documents(documents)
            st.success(f"✅ Created {len(document_chunks)} chunks")
            
            # Create embeddings
            embedding_model = OpenAIEmbeddings(
                api_key=api_key,
                base_url=api_base
            )
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                document_chunks,
                embedding_model
            )
            st.success("✅ Vector store created successfully!")
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5}
            )
            
            return vectorstore, retriever
            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None, None
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)


def generate_rag_response(user_input, retriever, client, k=5, max_tokens=500, temperature=0.3):
    """Generate response using RAG"""
    try:
        # Retrieve relevant documents - support multiple retriever API variants
        relevant_documents = None
        # Try common retriever APIs first
        try:
            if hasattr(retriever, 'get_relevant_documents'):
                relevant_documents = retriever.get_relevant_documents(user_input)
            elif hasattr(retriever, 'retrieve'):
                relevant_documents = retriever.retrieve(user_input)
        except Exception:
            relevant_documents = None

        # If retriever didn't yield documents, try using the stored vectorstore as a fallback
        if not relevant_documents:
            vs = None
            try:
                vs = st.session_state.get('vectorstore')
            except Exception:
                vs = None

            if vs is not None:
                # try a few vectorstore retrieval method names
                tried = []
                try:
                    if hasattr(vs, 'similarity_search'):
                        tried.append('similarity_search')
                        relevant_documents = vs.similarity_search(user_input, k)
                    elif hasattr(vs, 'similarity_search_with_score'):
                        tried.append('similarity_search_with_score')
                        res = vs.similarity_search_with_score(user_input, k)
                        relevant_documents = [d for d, s in res]
                    elif hasattr(vs, 'search'):
                        tried.append('search')
                        relevant_documents = vs.search(user_input, k)
                except Exception as e:
                    # fallthrough to diagnostics
                    relevant_documents = None

        # If still not found, return helpful diagnostic
        if not relevant_documents:
            available = ', '.join([a for a in dir(retriever) if not a.startswith('_')][:100])
            vs_name = type(st.session_state.get('vectorstore')).__name__ if st.session_state.get('vectorstore') else 'None'
            return f"Error generating response: no known retrieval method. Retriever type: {type(retriever).__name__}. VectorStore type: {vs_name}. Retriever attrs: {available}"
        context_list = [d.page_content for d in relevant_documents]
        context_for_query = ". ".join(context_list)
        
        # Format user message
        user_message = QNA_USER_MESSAGE_TEMPLATE.format(
            context=context_for_query,
            question=user_input
        )
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": QNA_SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🏥 Medical Knowledge Assistant</p>', unsafe_allow_html=True)
    st.markdown("**Powered by RAG | Grounded in Merck Manuals**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key or add it to config.json"
        )
        
        api_base_input = st.text_input(
            "API Base URL (Optional)",
            help="Leave empty for default OpenAI endpoint"
        )
        
        # Load credentials
        if api_key_input:
            api_key = api_key_input
            api_base = api_base_input if api_base_input else None
        else:
            api_key, api_base = load_api_credentials()
        
        if api_key:
            st.success("✅ API credentials loaded")
            if st.session_state.client is None:
                st.session_state.client = initialize_openai_client(api_key, api_base)
        else:
            st.warning("⚠️ Please provide API credentials")
        
        st.divider()
        
        # PDF Upload
        st.header("📄 Medical Manual Source")
        
        # Option to use pre-existing PDF or upload new one
        pdf_source = st.radio(
            "Choose PDF source:",
            ["Use existing PDF file", "Upload new PDF"],
            help="Select whether to use a PDF already in your directory or upload a new one"
        )
        
        if pdf_source == "Use existing PDF file":
            pdf_path = st.text_input(
                "PDF File Path",
                value="medical_diagnosis_manual.pdf",
                help="Enter the path to your medical PDF file"
            )
            
            if api_key and pdf_path:
                if st.button("Load PDF", type="primary"):
                    resolved = resolve_pdf_path(pdf_path)
                    if resolved:
                        with st.spinner("Processing PDF... This may take a few minutes."):
                            try:
                                # Load PDF
                                loader = PyMuPDFLoader(resolved)
                                documents = loader.load()
                                st.success(f"✅ Loaded {len(documents)} pages from PDF\nPath: {resolved}")
                                
                                # Split documents
                                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                    encoding_name='cl100k_base',
                                    chunk_size=512,
                                )
                                document_chunks = text_splitter.split_documents(documents)
                                st.success(f"✅ Created {len(document_chunks)} chunks")
                                
                                # Create embeddings
                                embedding_model = OpenAIEmbeddings(
                                    api_key=api_key,
                                    base_url=api_base
                                )
                                
                                # Create vector store
                                vectorstore = Chroma.from_documents(
                                    document_chunks,
                                    embedding_model
                                )
                                st.success("✅ Vector store created successfully!")
                                
                                # Create retriever
                                retriever = vectorstore.as_retriever(
                                    search_type='similarity',
                                    search_kwargs={'k': 5}
                                )
                                
                                st.session_state.vectorstore = vectorstore
                                st.session_state.retriever = retriever
                                
                            except Exception as e:
                                st.error(f"Error processing PDF: {e}")
                    else:
                        tried = [
                            pdf_path,
                            os.path.join(os.getcwd(), pdf_path),
                            os.path.join(os.path.dirname(__file__), pdf_path),
                        ]
                        st.error(f"❌ File not found: {pdf_path}. Tried: {tried}")
        
        else:  # Upload new PDF
            uploaded_file = st.file_uploader(
                "Upload PDF (e.g., Merck Manual)",
                type=['pdf'],
                help="Upload the medical reference manual to query"
            )
            
            if uploaded_file and api_key:
                if st.button("Process PDF", type="primary"):
                    vectorstore, retriever = process_pdf(uploaded_file, api_key, api_base)
                    if vectorstore and retriever:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = retriever
        
        st.divider()
        
        # Model parameters
        st.header("🎛️ Model Parameters")
        max_tokens = st.slider("Max Tokens", 100, 1000, 500)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        k_docs = st.slider("Retrieved Documents", 1, 10, 5)
        
        st.divider()
        
        # Status
        st.header("📊 Status")
        if st.session_state.retriever:
            st.success("✅ System Ready")
        else:
            st.info("ℹ️ Upload and process a PDF to begin")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Medical Question")
        
        # Sample questions
        with st.expander("📋 Sample Questions"):
            st.markdown("""
            - What is the protocol for managing sepsis in a critical care unit?
            - What are the common symptoms for appendicitis?
            - What are effective treatments for sudden patchy hair loss?
            - What treatments are recommended for traumatic brain injury?
            """)
        
        # Question input
        user_question = st.text_area(
            "Your Question:",
            height=100,
            placeholder="Enter your medical question here..."
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Generate response
        if ask_button and user_question:
            if not st.session_state.retriever:
                st.error("⚠️ Please upload and process a PDF first!")
            elif not st.session_state.client:
                st.error("⚠️ Please provide API credentials!")
            else:
                with st.spinner("Searching medical knowledge base..."):
                    response = generate_rag_response(
                        user_question,
                        st.session_state.retriever,
                        st.session_state.client,
                        k=k_docs,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response
                    })
        
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📜 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**Q{len(st.session_state.chat_history) - i}:** {chat['question']}")
                    
                    # Parse answer and source
                    answer_text = chat['answer']
                    if "Source:" in answer_text:
                        parts = answer_text.split("Source:")
                        st.markdown(f"**Answer:**\n{parts[0].replace('Answer:', '').strip()}")
                        st.markdown(f'<div class="source-citation"><strong>📚 Source:</strong> {parts[1].strip()}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Answer:**\n{answer_text}")
                    
                    st.divider()
    
    with col2:
        st.header("ℹ️ Information")
        
        st.info("""
        **How to use:**
        1. Enter your OpenAI API key in the sidebar
        2. Upload a medical reference PDF (e.g., Merck Manual)
        3. Click "Process PDF" to create the knowledge base
        4. Ask your medical questions!
        """)
        
        st.warning("""
        **⚠️ Disclaimer:**
        
        This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
        """)
        
        st.markdown("""
        **✨ Features:**
        - Evidence-based responses
        - Source citations
        - Grounded in medical literature
        - Retrieval-Augmented Generation (RAG)
        """)
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            **Technology Stack:**
            - LLM: GPT-4o-mini
            - Embeddings: OpenAI Ada-002
            - Vector DB: ChromaDB
            - Framework: LangChain
            - Chunking: 512 tokens with overlap
            - Retrieval: Similarity search (top-k)
            """)


if __name__ == "__main__":
    main()