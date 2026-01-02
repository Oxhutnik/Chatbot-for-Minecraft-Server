# 🧱 Chatbot for Minecraft Server

**Chatbot for Minecraft Server** is an intelligent, RAG-powered (Retrieval-Augmented Generation) assistant designed to help players on Minecraft servers. Built with **LangChain** and **Streamlit**, it answers user queries based on server documentation (PDF) while maintaining conversation context.

The system features a **modular multi-model architecture**, allowing seamless switching between **OpenAI (GPT-4o)** and **Google Gemini** models via a simple UI.

## 📂 Project Structure

The project follows a modular programming approach for better scalability, maintenance, and testing:

```text
├── app/
│   └── streamlit_app.py         # Main application interface (UI)
├── data/
│   └── minecraft_data.pdf       # Knowledge base (Server Documentation)
├── models/
│   ├── gpt_model.py             # OpenAI RAG implementation logic
│   └── gemini_model.py          # Google Gemini RAG implementation logic
├── tests/
│   ├── evaluate_metrics.py      # Automated testing script
│   └── test_data.json           # Ground-truth Q&A dataset
├── .env                         # API Keys (Not shared)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


🚀 Key Features

    RAG Architecture: Retrieves accurate information from a custom knowledge base (PDF guide) to answer specific server-related questions (IP, rules, commands).

    Multi-LLM Support: Supports both OpenAI (GPT-4o-mini) and Google (Gemini 2.5 Flash) with a modular backend.

    Conversational Memory: Remembers chat history to understand follow-up questions (e.g., "How do I claim?" -> "How do I delete it?").

    Smart Localization: Processes English documentation but enforces Turkish responses for local player support.

    Automated Testing: Includes a dedicated test module (tests/) to calculate Precision, Recall, and F1 Scores using an "LLM-as-a-Judge" approach.

🛠️ Tech Stack

    Python 3.11+

    LangChain: For chain orchestration and RAG flow.

    Streamlit: For the interactive web interface.

    ChromaDB: For vector storage and retrieval.

    LLMs: OpenAI GPT-4o & Google Gemini 2.5.
```

📦 Installation & Setup

    Clone the repository:
```bash

git clone [https://github.com/oxhutnik/Chatbot-for-Minecraft-Server.git](https://github.com/oxhutnik/Chatbot-for-Minecraft-Server.git)
cd Chatbot-for-Minecraft-Server
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Configure API Keys: Create a .env file in the root directory and add your API keys:
Code snippet

OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

Run the Application: (Note: Run from the root directory)
```bash
    streamlit run app/streamlit_app.py
```
📊 Testing & Evaluation

To benchmark the chatbot's accuracy against a ground-truth dataset, run the evaluation script located in the tests folder:
```bash
python tests/evaluate_metrics.py
```
This script runs a test set of 49 questions against both models and calculates the F1 Score to ensure response quality.
🏆 Benchmark Results

Below are the actual results from a test run using evaluate_metrics.py. In this specific scenario, Google Gemini achieved a higher accuracy in handling English-to-Turkish context switching.
Model	True Positive (TP)	False Positive (FP)	False Negative (FN)	F1 Score
Google (Gemini)	47	1	1	0.9792 👑
OpenAI (GPT-4o)	44	2	3	0.9462

> Note: Results represent a controlled test environment. The F1 Score reflects the model's ability to retrieve correct information from the PDF and generate accurate Turkish responses.
🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Developed by oxhutnik
