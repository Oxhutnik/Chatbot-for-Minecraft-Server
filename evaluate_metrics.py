import json
import os
import time
from dotenv import load_dotenv

# Models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import the RAG initializer from app.py
try:
    from app import initialize_rag_system
except ImportError:
    print("❌ ERROR: app.py not found.")
    exit()

load_dotenv()

# --- CONFIGURATION ---
JSON_FILE = "test_data.json"

# List the engines you want to test here.
# If you write ["google", "openai"], it tests both sequentially.
# If you write ["openai"], it only tests OpenAI.
ENGINES_TO_TEST = ["google", "openai"] 

# --- 1. DATA LOADING ---
def load_test_data(filepath):
    if not os.path.exists(filepath): return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return []

# --- 2. EVALUATOR SELECTION ---
def get_evaluator(provider):
    """
    Assigns an appropriate 'Evaluator' based on the engine being tested.
    GPT-4o is used when testing OpenAI, Gemini is used when testing Google.
    """
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        # Using Flash for Google to avoid hitting rate limits.
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def evaluate_answer(evaluator_llm, question, ground_truth, bot_answer):
    # --- PROMPT KEPT IN TURKISH AS REQUESTED ---
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

# --- 3. TEST ENGINE ---
def run_test_for_provider(provider_name, test_data):
    print(f"\n{'='*50}")
    print(f"🏁 STARTING TEST: {provider_name.upper()}")
    print(f"{'='*50}")
    
    # 1. Initialize RAG System for that provider
    # Passing parameters to the function in app.py
    rag_chain = initialize_rag_system(provider=provider_name)
    
    if not rag_chain:
        print(f"❌ {provider_name.upper()} could not be initialized (API Key or PDF missing).")
        return None

    # 2. Prepare Evaluator
    evaluator = get_evaluator(provider_name)
    
    results = {"TP": 0, "FP": 0, "FN": 0}
    
    print(f"🚀 Total {len(test_data)} questions will be answered...\n")

    for i, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{i+1}/{len(test_data)}] Question: {q}")
        
        # A) Bot Answer
        try:
            response = rag_chain.invoke({"input": q, "chat_history": []})
            bot_ans = response["answer"]
        except Exception as e:
            bot_ans = "ERROR"
            print(f"❌ Bot Error: {e}")

        # B) Evaluator Decision
        decision = evaluate_answer(evaluator, q, gt, bot_ans)
        
        if "TRUE_POSITIVE" in decision: results["TP"] += 1
        elif "FALSE_POSITIVE" in decision: results["FP"] += 1
        else: results["FN"] += 1
        
        print(f"   🤖 Answer: {bot_ans[:80]}...") # Show beginning of the answer
        print(f"   ⚖️ Decision: {decision}")
        
        # C) Rate Limiting Wait
        if provider_name == "google":
            time.sleep(10) # Wait required for Gemini Free Tier
        else:
            time.sleep(0.5) # Short wait sufficient for OpenAI

    # Calculate Statistics
    tp, fp, fn = results["TP"], results["FP"], results["FN"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"TP": tp, "FP": fp, "FN": fn, "F1": f1}

# --- 4. MAIN PROGRAM ---
def main():
    test_data = load_test_data(JSON_FILE)
    if not test_data:
        print(f"❌ '{JSON_FILE}' not found or empty!")
        return

    final_results = {}

    for provider in ENGINES_TO_TEST:
        score = run_test_for_provider(provider, test_data)
        if score:
            final_results[provider] = score

    # --- FINAL TABLE ---
    if final_results:
        print("\n\n" + "#"*60)
        print(f"{'🏆 COMPARISON TABLE 🏆':^60}")
        print("#"*60)
        print(f"{'MODEL':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'F1 SCORE':<10}")
        print("-" * 60)
        
        for p, s in final_results.items():
            print(f"{p.upper():<12} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {s['F1']:.4f}")
        print("#"*60 + "\n")

if __name__ == "__main__":
    main()