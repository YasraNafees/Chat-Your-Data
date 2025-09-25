# PDF Question Answering Chatbot  

This project is a **PDF-based Question Answering Chatbot** built using **LangChain**, **FAISS**, **Hugging Face Embeddings**, and **OpenRouter LLMs**.  
It allows you to upload a PDF (e.g., *AI Foundations of Computational Agents*) and ask natural language questions.  
The system retrieves relevant chunks from the PDF and uses a Large Language Model to answer.  

---

## 🚀 Features  
- Load and index any PDF into a vector database.  
- Uses **FAISS** for fast similarity search.  
- Embeds text with **sentence-transformers/all-MiniLM-L6-v2**.  
- Queries handled by **DeepSeek Chat (via OpenRouter)**.  
- Streaming responses for real-time answers.  
- Supports caching of PDF embeddings for efficiency.  

---

## 📂 Project Structure  
```
.
├── app.py              # Entry point (frontend or API server)
├── backend.py          # Core logic: PDF loading, splitting, embeddings, vector store, LLM pipeline
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys, base URLs)
├── venv/               # Virtual environment
├── .indices/           # Cached vectorstore indices
├── __pycache__/        # Python cache
└── AI Foundations of Computational Agents 3rd Ed.pdf  # Example PDF
```

---

## ⚙️ Setup Instructions  

### 1. Clone the repository  
```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create a virtual environment  
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies  
```bash
pip install -r requirements.txt
```

### 4. Add environment variables  
Create a **.env** file in the project root:  
```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

### 5. Run the app  
If `app.py` starts the chatbot:  
```bash
python app.py
```

---

## 🧠 How It Works  
1. **PDF Loader** → Splits PDF into chunks.  
2. **Embeddings** → Converts chunks into vectors using Hugging Face.  
3. **Vector Store (FAISS)** → Stores and retrieves relevant chunks.  
4. **LLM (DeepSeek via OpenRouter)** → Answers user queries with context.  
5. **Streaming** → Sends tokens back as they are generated.  

---

## 📌 Example Usage  
```python
from backend import setup_pipeline_and_query

for token in setup_pipeline_and_query(
    pdf_path="AI Foundations of Computational Agents 3rd Ed.pdf",
    question="What is an intelligent agent?"
):
    print(token, end="")
```

---

## 📜 License  
MIT License – feel free to use and modify.  
