import os
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 1. Configuration of the models
# We use a multilingual embedding model to handle both French (IRFA) and English (Quant) documents efficiently.
# 'paraphrase-multilingual-MiniLM-L12-v2' is fast and effective for this mix.
print("Loading embedding model (this may take a minute the first time)...")
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# We set the global settings for LlamaIndex
Settings.embed_model = embed_model
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)

# Define directories
DATA_DIR = "./data"
STORAGE_DIR = "./storage"

def create_knowledge_base():
    # Check if data folder exists and is not empty
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"Error: The directory '{DATA_DIR}' does not exist or is empty.")
        print("Please create it and add your PDF files.")
        return None

    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        print("Creating new index from documents...")
        
        # Load documents from the data folder
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        print(f"Loaded {len(documents)} document pages.")

        # Create the index (this is where the 'magic' happens: converting text to vectors)
        index = VectorStoreIndex.from_documents(documents)

        # Persist (save) the index to disk so we don't have to rebuild it every time
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"Index successfully saved to '{STORAGE_DIR}'.")
    else:
        print("Loading existing index from storage...")
        # Load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
        print("Index loaded.")

    return index

if __name__ == "__main__":
    index = create_knowledge_base()
    
    if index:
        print("\n--- Test Query on your Data ---")
        # Let's verify it works with a simple query engine
        query_engine = index.as_query_engine()
        # You can change this question to something specific to your courses
        response = query_engine.query("Summarize the main topics covered in the documents provided.")
        print(response)