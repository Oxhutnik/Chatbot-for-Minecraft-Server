# 🧱 Chatbot for Minecraft Server

**Chatbot for Minecraft Server** is an intelligent, RAG-powered (Retrieval-Augmented Generation) assistant designed to help players on Minecraft servers. Built with **LangChain** and **Streamlit**, it answers user queries based on server documentation (PDF) while maintaining conversation context.

The system features a **multi-model architecture**, allowing seamless switching between **OpenAI (GPT-4o)** and **Google Gemini** models via a simple UI.

## 🚀 Key Features

* **RAG Architecture:** Retrieves accurate information from a custom knowledge base (PDF guide) to answer specific server-related questions (IP, rules, commands).
* **Multi-LLM Support:** Supports both **OpenAI (GPT-4o-mini)** and **Google (Gemini 1.5 Flash)**.
* **Conversational Memory:** Remembers chat history to understand follow-up questions (e.g., "How do I claim?" -> "How do I delete **it**?").
* **Smart Localization:** Processes English documentation but enforces Turkish responses for local player support.
* **Performance Evaluation:** Includes a custom `evaluate_metrics.py` script to calculate Precision, Recall, and F1 Scores using an "LLM-as-a-Judge" approach.

## 🛠️ Tech Stack

* **Python 3.11+**
* **LangChain:** For chain orchestration and RAG flow.
* **Streamlit:** For the interactive web interface.
* **ChromaDB:** For vector storage and retrieval.
* **LLMs:** OpenAI GPT-4o & Google Gemini 1.5.

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/oxhutnik/Chatbot-for-Minecraft-Server.git](https://github.com/oxhutnik/Chatbot-for-Minecraft-Server.git)
    cd Chatbot-for-Minecraft-Server
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    GOOGLE_API_KEY=your_google_api_key_here
    ```

4.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 📊 Testing & Evaluation

To benchmark the chatbot's accuracy against a ground-truth dataset:

```bash
python evaluate_metrics.py

This script runs a test set of 50 questions against both models and calculates the F1 Score to ensure response quality.
🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Developed by oxhutnik
