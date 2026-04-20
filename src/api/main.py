import os
import json
from llama_cpp import Llama
from fastapi import FastAPI
from pydantic import BaseModel
from src.api.redis_cache import get_semantic_match, set_semantic_match
from src.utils.logger import trace_execution

app = FastAPI(title="Yelp-Style GenAI Query API")

# --- Initialize Local Generative LLM ---
# Using the absolute path is safer on shared servers
MODEL_PATH = "/home/amanr.mds2024/Developer/AML/GenAI_proj/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH, 
    n_gpu_layers=-1, # Offload to your 2080 Ti
    n_ctx=512, 
    verbose=False    # Set back to False now that we know it loads
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    extracted_entities: dict
    source: str

SYSTEM_PROMPT = """Extract 'topic', 'location', and 'time' from the user query.
Return ONLY valid JSON. No preamble, no backticks.
Focus on the primary business type or service requested as the topic.
Example: {"topic": "sushi", "location": "nyc", "time": "today"}"""

@app.post("/parse_query", response_model=QueryResponse)
@trace_execution("parse_query_endpoint")
def parse_query(request: QueryRequest):
    # 1. Check Semantic Cache
    cached_result = get_semantic_match(request.query)
    if cached_result:
        return QueryResponse(
            query=request.query,
            extracted_entities=json.loads(cached_result),
            source="redis_semantic_cache"
        )
    
    # 2. Generative Inference (Optimized for Llama 3.1 Instruct)
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{request.query}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    output = llm(prompt, max_tokens=150, stop=["<|eot_id|>", "\n\n"], echo=False)
    raw_text = output['choices'][0]['text'].strip()
    
    # 3. Robust JSON Parsing (Strips extra conversational text if present)
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = raw_text[start_idx:end_idx]
            generated_entities = json.loads(json_str)
        else:
            raise ValueError("No JSON found")
    except Exception as e:
        # Emergency Fallback
        generated_entities = {
            "topic": request.query, 
            "location": "unknown", 
            "parsing": "cleaned_fallback"
        }

    # 4. Store in Cache
    set_semantic_match(request.query, json.dumps(generated_entities))
    
    return QueryResponse(
        query=request.query,
        extracted_entities=generated_entities,
        source="local_llama_inference"
    )