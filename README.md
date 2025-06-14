# 🤖 Multiagent-Communication

Welcome to the **Multi-Agent Communication System** – a modular, scalable, AI-powered architecture designed for intelligent task coordination using CHAT, RAG, SUPERVISOR, and AIML agents. This system powers enhanced document processing, medical diagnostics, and multimodal retrieval using local LLMs, image models, and vector-based retrieval.

---

## 🏗️ Project Structure

```
VECTORISED_MULTIAGENT/
├── agents/
│   ├── AIML/
│   │   ├── models/
│   │   │   └── model_loader.py
│   │   ├── processors/
│   │   │   ├── image_processor.py
│   │   │   ├── report_generator.py
│   │   │   └── stage_classifier.py
│   │   ├── templates/
│   │   │   └── Medical_Report_Sample.pdf
│   │   ├── utils/
│   │   │   ├── aiml_agent.py
│   │   │   ├── image_utils.py
│   │   │   └── pdf_utils.py
│   │   └── __init__.py
│
│   ├── CHAT/
│   │   ├── generators/
│   │   │   ├── llm_generator.py
│   │   │   └── structured_generator.py
│   │   ├── models/
│   │   │   └── model_loader.py
│   │   ├── processors/
│   │   │   ├── chat_agent.py
│   │   │   ├── chat_agent_api.py
│   │   │   ├── query_analyzer.py
│   │   │   ├── rag_processor.py
│   │   │   └── response_formatter.py
│   │   └── __init__.py
│
│   ├── RAG/
│   │   ├── data_structures.py
│   │   ├── rag_agent.py
│   │   ├── retrieval.py
│   │   ├── text_processing.py
│   │   └── vectorization.py
│   │   └── __init__.py
│
│   ├── SUPERVISOR/
│   │   ├── components/
│   │   │   ├── error_handler.py
│   │   │   ├── performance_monitor.py
│   │   │   ├── session_manager.py
│   │   │   ├── system_health.py
│   │   │   └── __init__.py
│   │   ├── core/
│   │   │   ├── enhanced_supervisor.py
│   │   │   ├── message_analysis.py
│   │   │   ├── supervisor_config.py
│   │   │   └── __init__.py
│   │   ├── processors/
│   │   │   ├── chat_processor.py
│   │   │   ├── rag_processor.py
│   │   │   └── system_command.py
│   │   │   └── __init__.py
│   │   ├── utils/
│   │   │   ├── display_utils.py
│   │   │   ├── response_formatter.py
│   │   │   ├── time_utils.py
│   │   │   └── __init__.py
│   │   └── supervisor_agent.py
│   │   └── __init__.py
│
├── knowledge_base/
│   ├── generated_reports/
│   ├── graph/
│   ├── papers/
│   ├── reports/
│   └── vectors/
│
├── models/
├── old_agents/
├── shared_memory/
│   ├── simple_memory.py
│   └── __init__.py
├── venv/
├── .env
├── .gitignore
├── app.py
├── main.py
└── requirements.txt
```

---

## 🧠 Key Features

* 🗣️ **Agent-to-Agent Communication** (CHAT ↔ RAG ↔ SUPERVISOR)
* 📄 **Medical Report Generator** with dynamic MRI scan embedding
* 🧠 **AI Reasoning with Local SLMs** like Phi-3 Mini, Qwen2.5, TinyLlama
* 🔍 **Multimodal Retrieval** (text + image) using CLIP + MiniLM embeddings
* 🧾 **PDF Reporting** with clinical summaries, exam findings, diagnosis
* 📊 **FAISS-based vector search** over knowledge base
* 📦 Fully offline + streamlit powered app

---

## 🩺 Medical Report Generation

> The `AIML/` agent uses OCR + Stage Classification + PDF templating to generate beautiful, doctor-style medical reports.

Example Features:

* Patient details table
* Clinical summary + diagnosis
* Treatment plan
* MRI scan image placeholder (auto-filled)
* Doctor’s signature

---

## 🚀 Running the System

```python
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 📦 Dependencies

Includes:

* `transformers`, `torch`, `sentence-transformers`, `faiss-cpu`
* `pydicom`, `opencv-python`, `PyMuPDF`, `PyPDF2`
* `streamlit`, `easyocr`, `tqdm`, `Pillow`

---

## 📂 Knowledge Base Format

```
knowledge_base/
├── reports/              # Medical report text chunks
├── papers/               # Research papers (PDFs)
├── graph/                # Graph data (e.g., knowledge graphs)
├── vectors/              # FAISS vector DBs
```

---

## 👨‍💻 Maintained By

Built with 🧠 by [@Sagar4276](https://github.com/Sagar4276)

---

## 📝 License

MIT License (or specify your own)
