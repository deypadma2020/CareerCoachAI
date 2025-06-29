# CareerCoachAI
An AI-powered interview coaching chatbot that remembers past conversations and gives smart, context-aware follow-up answers using Claude 3.5 and Cohere embeddings.

---

````markdown
# CareerCoachAI

**CareerCoachAI** is an intelligent AI-powered chatbot that helps you prepare for technical job interviews. It remembers your past questions, retrieves relevant previous answers using semantic similarity, and gives you context-aware follow-up responses—just like a real interview coach.

---

## 🚀 Features

- 🧠 **Memory-based Interview Coaching**  
  Stores and recalls previous conversations for smarter interactions.

- 🤖 **Powered by Claude 3.5 Sonnet**  
  Uses the latest Anthropic model via Langchain for accurate and natural responses.

- 🔍 **Semantic Recall with Cohere Embeddings**  
  Embeds and compares user queries for smart similarity-based memory lookup.

- 💬 **Streamlit Frontend**  
  Clean, interactive interface for easy chat experience.

- 🔧 **Modular Python Architecture**  
  Clean separation of UI, logic, memory, and embeddings for easy extension.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CareerCoachAI.git
cd CareerCoachAI
````

### 2. Setup `.env` File

Create a `.env` file in the root directory with the following:

```
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run streamlit_app.py
```

---

## 🗂️ Project Structure

```
CareerCoachAI/
├── streamlit_app.py         # Streamlit app interface
├── chatbot.py               # Chat logic and prompt invocation
├── chat_utils.py            # Load/save chat history
├── embedding_utils.py       # Cohere embedding + similarity matching
├── history.json             # Persistent chat history
├── embeddings.json          # Embedded Q&A for similarity recall
├── requirements.txt
├── README.md
└── .env                     # API keys (not tracked by Git)
```

---

## 🔮 Future Enhancements

* Use FAISS or ChromaDB for scalable vector search
* Add resume-based Q\&A and context awareness
* Topic tagging for strengths/weakness identification
* Session-level tracking or multi-user support
* Dashboard for interview prep analytics

---

## 🧠 Built With

* [Streamlit](https://streamlit.io/)
* [Langchain](https://www.langchain.com/)
* [Anthropic Claude 3.5 Sonnet](https://www.anthropic.com/index/claude)
* [Cohere Embeddings](https://cohere.com/)
* [Scikit-learn](https://scikit-learn.org/)

---

## 🛡 License

This project is licensed under the MIT License.

---

## 👤 Author

**\Padma Dey**
