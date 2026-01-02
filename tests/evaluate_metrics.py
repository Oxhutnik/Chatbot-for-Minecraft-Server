import json
import os
import sys
import time
from dotenv import load_dotenv

# --- YOL AYARLAMASI (Path Fix) ---
# Test dosyası 'tests' klasöründe olduğu için, 
# bir üst dizine çıkıp 'models' ve 'data' klasörlerini görmesini sağlıyoruz.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- İMPORTLAR ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Direkt modellerden import ediyoruz (Streamlit arayüzüne bulaşmadan)
from models.gpt_model import initialize_openai_rag
from models.gemini_model import initialize_gemini_rag

load_dotenv(os.path.join(parent_dir, ".env"))

# --- AYARLAR ---
# JSON dosyası artık test dosyasıyla aynı klasörde
JSON_DOSYASI = os.path.join(current_dir, "test_data.json")
TEST_EDILECEK_MOTORLAR = ["google", "openai"]

# --- 1. VERİ HAZIRLIĞI (PDF Yükleme) ---
def prepare_docs():
    """PDF'i yükler ve parçalar (Chunking)."""
    pdf_path = os.path.join(parent_dir, "data", "minecraft_data.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"❌ HATA: PDF bulunamadı: {pdf_path}")
        return None
        
    try:
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        return docs
    except Exception as e:
        print(f"❌ PDF Okuma Hatası: {e}")
        return None

# --- 2. JSON YÜKLEME ---
def load_test_data(filepath):
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return []

# --- 3. HAKEM (EVALUATOR) ---
def get_evaluator(provider):
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def evaluate_answer(evaluator_llm, question, ground_truth, bot_answer):
    prompt = ChatPromptTemplate.from_template(
        """
        Sen adil bir öğretmensin. 
        Soru: {question}
        Referans Cevap: {ground_truth}
        Öğrenci Cevabı: {bot_answer}
        
        Lütfen öğrencinin cevabını değerlendir ve şu 3 kategoriden SADECE BİRİNİ yaz:
        TRUE_POSITIVE : Cevap doğru ve yeterli.
        FALSE_POSITIVE : Cevap yanlış veya uydurma (Hallucination).
        FALSE_NEGATIVE : Cevap yok, "bilmiyorum" denmiş veya çok eksik.
        
        Karar (Sadece kategoriyi yaz):
        """
    )
    chain = prompt | evaluator_llm
    try:
        return chain.invoke({
            "question": question, 
            "ground_truth": ground_truth, 
            "bot_answer": bot_answer
        }).content.strip().upper()
    except:
        return "ERROR"

# --- 4. TEST MOTORU ---
def run_test_for_provider(provider_name, test_data, docs):
    print(f"\n{'='*50}")
    print(f"🏁 TEST BAŞLATILIYOR: {provider_name.upper()}")
    print(f"{'='*50}")
    
    # RAG Zincirini Kur
    rag_chain = None
    if provider_name == "openai":
        rag_chain = initialize_openai_rag(docs)
    elif provider_name == "google":
        rag_chain = initialize_gemini_rag(docs)

    if not rag_chain:
        print("❌ Zincir kurulamadı.")
        return None

    # Hakemi Hazırla
    evaluator = get_evaluator(provider_name)
    results = {"TP": 0, "FP": 0, "FN": 0}
    
    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{i+1}/{len(test_data)}] Soru: {q}")
        
        # Bot Cevabı
        try:
            response = rag_chain.invoke({"input": q, "chat_history": []})
            bot_ans = response["answer"]
        except Exception as e:
            bot_ans = "ERROR"
            print(f"❌ Hata: {e}")

        # Hakem Kararı
        decision = evaluate_answer(evaluator, q, gt, bot_ans)
        
        if "TRUE_POSITIVE" in decision: results["TP"] += 1
        elif "FALSE_POSITIVE" in decision: results["FP"] += 1
        else: results["FN"] += 1
        
        print(f"   🤖 Cevap: {bot_ans[:60]}...") 
        print(f"   ⚖️ Karar: {decision}")
        
        if provider_name == "google": time.sleep(4) # Rate limit koruması
        else: time.sleep(0.5)

    # İstatistik
    tp, fp, fn = results["TP"], results["FP"], results["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"TP": tp, "FP": fp, "FN": fn, "F1": f1}

# --- 5. ANA PROGRAM ---
def main():
    # Önce veriyi bir kere yükle (her model için tekrar yüklemeye gerek yok)
    print("📂 PDF Verisi Yükleniyor...")
    docs = prepare_docs()
    if not docs: return

    test_data = load_test_data(JSON_DOSYASI)
    if not test_data:
        print(f"❌ '{JSON_DOSYASI}' bulunamadı!")
        return

    final_results = {}

    for provider in TEST_EDILECEK_MOTORLAR:
        score = run_test_for_provider(provider, test_data, docs)
        if score:
            final_results[provider] = score

    # --- FİNAL TABLOSU ---
    if final_results:
        print("\n\n" + "#"*60)
        print(f"{'🏆 KARŞILAŞTIRMA TABLOSU 🏆':^60}")
        print("#"*60)
        print(f"{'MODEL':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'F1 SKORU':<10}")
        print("-" * 60)
        
        for p, s in final_results.items():
            print(f"{p.upper():<12} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {s['F1']:.4f}")
        print("#"*60 + "\n")

if __name__ == "__main__":
    main()
