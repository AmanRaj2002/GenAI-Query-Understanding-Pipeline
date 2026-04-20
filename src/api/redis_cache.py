import yaml
from pathlib import Path
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

# Load config safely
def load_config():
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

# Initialize the vectorizer (converts text to high-dimensional embeddings)
vectorizer = HFTextVectorizer(config['cache']['embedding_model'])

# Initialize Semantic Cache backed by Redis Vector Search
# This allows queries like "how to reset password" and "forgot password steps" to share a cache hit
semantic_cache = SemanticCache(
    name=config['cache']['index_name'],
    redis_url=config['cache']['redis_url'],
    distance_threshold=config['cache']['distance_threshold'],
    vectorizer=vectorizer
)

def get_semantic_match(query: str):
    """
    Embeds the query and searches the vector database for a semantically similar cached intent.
    """
    # Returns the cached response if the cosine distance is below the threshold (e.g., 0.15)
    if results := semantic_cache.check(prompt=query):
        return results[0]["response"]
    return None

def set_semantic_match(query: str, parsed_intent: str):
    """
    Vectorizes the new query and stores it alongside the LLM's parsed intent.
    """
    semantic_cache.store(prompt=query, response=parsed_intent)