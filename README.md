# Enterprise Query Understanding Pipeline 

An end-to-end NLP pipeline for Yelp-style enterprise search, utilizing high-performance local LLMs and semantic caching.
This code base closely follows the following blog's methodology:
Part II: modern
LLMs/GenAI: e.g. Yelp Search Query Understanding (https://engineeringblog.yelp.com/2025/02/search-query-understanding-with-LLMs.html)

##  Tech Stack
* **LLM:** Llama 3.1 8B Instruct (GGUF Quantized)
* **Inference:** `llama-cpp-python` with Dual NVIDIA RTX 2080 Ti (CUDA offloading)
* **Backend:** FastAPI
* **Cache:** Redis Stack (Semantic Caching with Sentence-Transformers)
* **Tracking:** MLflow (Experiment tracking and intent logging)
* **UI:** Streamlit

##  Architecture
1. **Semantic Cache:** Matches user queries against Redis vector store to provide 5ms responses for repeated intents.
2. **Generative Extraction:** For new queries, Llama 3.1 extracts JSON entities (topic, location, time) using custom Instruct templates.
3. **Hardware Acceleration:** Full GPU offloading across multi-GPU setup.
4. **Monitoring:** All inference traces are logged to an MLflow tracking server.

##  How to Run
1. Run `python start_stack.py` to orchestrate all services.
2. Access the UI at `http://localhost:8501`.
