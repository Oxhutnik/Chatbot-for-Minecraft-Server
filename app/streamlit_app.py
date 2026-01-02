import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage

# Python'un ana dizindeki 'models' klasörünü görmesi için yol ayarı
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Modelleri import ediyoruz
from models.gpt_model import initialize_openai_rag
from models.gemini_model import initialize_gemini_rag

# .env dosyasını bir üst dizinden yükle
load_dotenv(os.path.join(parent_dir, ".env"))

# --- AYARLAR ---
SELECTED_MODEL = "openai"  # "openai" veya "google"

# Sayfa Ayarı
if SELECTED_MODEL == "openai":
    baslik = "🧱 ChatbotCraft Assistant (GPT-4o)"
else:
    baslik = "🧱 ChatbotCraft Assistant (Gemini)"

st.set_page_config(page_title=baslik)
st.title(baslik)

# --- VERİ YÜKLEME (DATA LOADER) ---
@st.cache_resource
def load_and_split_data():
    # PDF yolu artık data klasörünün içinde
    file_path = os.path.join(parent_dir, "data", "minecraft_data.pdf")
    
    if not os.path.exists(file_path):
        st.error(f"Dosya bulunamadı: {file_path}")
        return None

    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        return docs
    except Exception as e:
        st.error(f"Veri Yükleme Hatası: {e}")
        return None

# --- SİSTEM BAŞLATICI ---
@st.cache_resource
def get_rag_chain(model_name):
    docs = load_and_split_data()
    if not docs:
        return None

    print(f"🔄 Model Yükleniyor: {model_name.upper()}")
    
    if model_name == "openai":
        return initialize_openai_rag(docs)
    elif model_name == "google":
        return initialize_gemini_rag(docs)
    else:
        st.error("Geçersiz model seçimi!")
        return None

# --- UYGULAMA AKIŞI ---
if __name__ == "__main__":
    
    if 'rag_chain' not in st.session_state:
        with st.spinner(f"{SELECTED_MODEL.upper()} motoru hazırlanıyor..."):
            st.session_state.rag_chain = get_rag_chain(SELECTED_MODEL)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Sorunuzu buraya yazın..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        if st.session_state.rag_chain:
            chat_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.chat_message("assistant"):
                with st.spinner("Cevap yazılıyor..."):
                    try:
                        response = st.session_state.rag_chain.invoke({
                            "input": user_input,
                            "chat_history": chat_history
                        })
                        answer = response["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Hata: {e}")
