import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def initialize_gemini_rag(docs):
    """Google Gemini tabanlı RAG zincirini kurar."""
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY eksik!")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Vektör Veritabanı
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        collection_name="minecraft_google"
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Promptlar (GPT ile aynı mantık ama ayrı dosyada)
    contextualize_q_system_prompt = (
        "Sohbet geçmişi ve son soru verildiğinde, eğer soru geçmişe atıfta bulunuyorsa "
        "soruyu tek başına anlaşılır hale getir. Cevaplama, sadece soruyu düzelt."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
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

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

    return rag_chain
