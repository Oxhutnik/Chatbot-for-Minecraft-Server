import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- CONFIGURATION ---
SELECTED_MODEL = "openai"  # Options: "openai" or "google"


load_dotenv()

# Set page title and icon based on the selected model
# GÜNCELLEME: İsim ChatbotCraft yapıldı
if SELECTED_MODEL == "openai":
    page_title = "🧱 ChatbotCraft Assistant (GPT-4o)"
else:
    page_title = "🧱 ChatbotCraft Assistant (Gemini)"

st.set_page_config(page_title=page_title)
st.title(f"{page_title}")

# --- RAG SYSTEM ---
@st.cache_resource
def initialize_rag_system(provider):
    print(f"🔄 System Initializing... Selected Engine: {provider.upper()}")
    
    file_path = "minecraft_data.pdf"
    if not os.path.exists(file_path):
        st.error("PDF file not found!")
        return None

    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # MODEL SELECTION
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY missing!")
            st.stop()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY missing!")
            st.stop()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # Note: Ensure you have access to 2.5, otherwise use "gemini-1.5-flash"
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Create a collection with a different name for each provider (google/openai)
    # This prevents dimension mismatch errors (3072 vs 1536).
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        collection_name=f"minecraft_{provider}" 
    )    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # PROMPT SETTINGS
    
    # Contextualization Prompt (Reformulates the question based on history)
    contextualize_q_system_prompt = (
        "Sohbet geçmişi ve son soru verildiğinde, eğer soru geçmişe atıfta bulunuyorsa "
        "soruyu tek başına anlaşılır hale getir. Cevaplama, sadece soruyu düzelt."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA Prompt (Instructs the bot to answer based on context)
    # GÜNCELLEME: Prompt içinde sunucu ismi ChatbotCraft olarak değiştirildi.
    system_prompt = (
        "Sen ChatbotCraft Minecraft sunucusunun yardımsever asistanısın. "
        "Aşağıdaki bağlamı (context) kullanarak soruları cevapla. "
        "Bağlam İngilizce olabilir ama sen her zaman **TÜRKÇE** cevap vermelisin. "
        "Bilmiyorsan 'Bilmiyorum' de."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)
    
    return rag_chain

# --- APPLICATION FLOW ---
if __name__ == "__main__":
    
    # Initialize the chain (Run only once)
    if 'rag_chain' not in st.session_state:
        with st.spinner(f"{SELECTED_MODEL.upper()} engine warming up..."):
            st.session_state.rag_chain = initialize_rag_system(SELECTED_MODEL)

    # Message History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Type your question here..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        if st.session_state.rag_chain:
            chat_history = []
            # Prepare chat history for LangChain
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        response = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": chat_history
                        })
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")