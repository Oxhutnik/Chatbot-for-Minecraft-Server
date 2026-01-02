# 🧱 Chatbot for Minecraft Server

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Chatbot for Minecraft Server** is an intelligent, RAG-powered (Retrieval-Augmented Generation) assistant designed to help players on Minecraft servers. Built with **LangChain** and **Streamlit**, it answers user queries based on server documentation (PDF) while maintaining conversation context.

The system features a **modular multi-model architecture**, allowing seamless switching between **OpenAI (GPT-4o)** and **Google Gemini** models via a simple UI.

---

## 📑 Table of Contents

- [Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation--setup)
- [Usage](#-usage)
- [Testing](#-testing--evaluation)
- [Benchmark Results](#-benchmark-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Key Features

✅ **RAG Architecture**: Retrieves accurate information from a custom knowledge base (PDF guide) to answer specific server-related questions (IP, rules, commands).

✅ **Multi-LLM Support**: Supports both OpenAI (GPT-4o-mini) and Google (Gemini 2.5 Flash) with a modular backend.

✅ **Conversational Memory**: Remembers chat history to understand follow-up questions (e.g., "How do I claim?" → "How do I delete it?").

✅ **Smart Localization**: Processes English documentation but enforces Turkish responses for local player support.

✅ **Automated Testing**: Includes a dedicated test module (`tests/`) to calculate Precision, Recall, and F1 Scores using an "LLM-as-a-Judge" approach.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core programming language |
| **LangChain** | Chain orchestration and RAG flow |
| **Streamlit** | Interactive web interface |
| **ChromaDB** | Vector storage and retrieval |
| **OpenAI GPT-4o** | Primary LLM option |
| **Google Gemini 2.5** | Alternative LLM option |

---

## 📂 Project Structure

The project follows a modular programming approach for better scalability, maintenance, and testing:

```
Chatbot-for-Minecraft-Server/
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
├── LICENSE                      # Project license
└── README.md                    # Project documentation
```

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- OpenAI API Key
- Google API Key (for Gemini)

### Step 1: Clone the Repository

```bash
git clone https://github.com/oxhutnik/Chatbot-for-Minecraft-Server.git
cd Chatbot-for-Minecraft-Server
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env` file in the root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

> ⚠️ **Important**: Never commit your `.env` file to version control. Add it to `.gitignore`.

### Step 4: Run the Application

Execute from the root directory:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 💬 Usage

1. **Select a Model**: Choose between OpenAI (GPT-4o) or Google Gemini from the sidebar.
2. **Ask Questions**: Type your Minecraft server-related questions in Turkish or English.
3. **Get Answers**: The chatbot retrieves information from the PDF documentation and responds in Turkish.
4. **Conversational Context**: Ask follow-up questions naturally - the bot remembers your conversation history.

### Example Queries

- "Sunucu IP adresi nedir?" (What is the server IP address?)
- "Nasıl arazi claim edebilirim?" (How can I claim land?)
- "Hangi komutlar kullanılabilir?" (What commands are available?)

---

## 📊 Testing & Evaluation

To benchmark the chatbot's accuracy against a ground-truth dataset, run the evaluation script:

```bash
python tests/evaluate_metrics.py
```

This script:
- Runs a test set of **49 questions** against both models
- Uses an "LLM-as-a-Judge" approach for evaluation
- Calculates **Precision**, **Recall**, and **F1 Score**
- Generates a detailed performance report

---

## 🏆 Benchmark Results

Below are the actual results from a test run using `evaluate_metrics.py`. In this specific scenario, Google Gemini achieved higher accuracy in handling English-to-Turkish context switching.

| Model | True Positive (TP) | False Positive (FP) | False Negative (FN) | F1 Score |
|-------|-------------------|---------------------|---------------------|----------|
| **Google (Gemini)** 👑 | 47 | 1 | 1 | **0.9792** |
| **OpenAI (GPT-4o)** | 44 | 2 | 3 | **0.9462** |

> **Note**: Results represent a controlled test environment. The F1 Score reflects the model's ability to retrieve correct information from the PDF and generate accurate Turkish responses.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Developed by [oxhutnik](https://github.com/oxhutnik)**

---

## 📧 Contact & Support

If you encounter any issues or have questions:

- Open an [Issue](https://github.com/oxhutnik/Chatbot-for-Minecraft-Server/issues)
- Start a [Discussion](https://github.com/oxhutnik/Chatbot-for-Minecraft-Server/discussions)

---

## ⭐ Show Your Support

If you find this project useful, please consider giving it a star on GitHub! It helps others discover the project.

[![GitHub stars](https://img.shields.io/github/stars/oxhutnik/Chatbot-for-Minecraft-Server?style=social)](https://github.com/oxhutnik/Chatbot-for-Minecraft-Server/stargazers)

---

**Happy Gaming! 🎮**
