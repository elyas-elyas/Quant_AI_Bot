# Local RAG Assistant: IRFA & Quantitative Finance

A specialized **Retrieval-Augmented Generation (RAG)** chatbot designed for Quantitative Finance. This project leverages local LLMs to provide context-aware answers based on private course materials and academic papers, running entirely offline for privacy and performance.


---

## Project Objective

Provide quantitative researchers and analysts with a private AI assistant capable of digesting hundreds of technical documents (PDFs) to answer complex financial questions with precise source citations:

1. **Knowledge Ingestion (`ingest_data.py`)**: A vectorization pipeline that converts academic courses and research papers into high-dimensional embeddings using multilingual models.
2. **Interactive Assistant (`app.py`)**: A Streamlit-based chat interface that retrieves relevant context, generates answers using **Llama 3.1**, and cites the exact document and page number used.

---

## Key Features

| Feature | Description | Technical Component |
|---------|-------------|---------------------|
| **Retrieval-Augmented Generation** | Combines the reasoning power of LLMs with specific, private knowledge bases. | RAG Engine |
| **Precise Source Citation** | Every answer includes a "Sources" dropdown indicating the filename and page number. | Metadata Extraction |
| **Multilingual Support** | Optimized to handle queries and documents in both **French** (IRFA Courses) and **English** (Quant Papers). | Embeddings |
| **Local Privacy** | No data leaves the machine. Uses local Ollama instance and local vector storage. | Ollama / ChromaDB |
| **Persistent Memory** | The vector index is saved to disk (`/storage`) to avoid re-indexing documents at every restart. | Vector Store |
| **Context Awareness** | Maintains conversation history to answer follow-up questions ("Explain more about that"). | Chat Engine |

---

## Methodology

### 1. Vector Embeddings (Multilingual)

* **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **Purpose**: Converts text from PDFs into numerical vectors. This specific model was chosen for its ability to map French and English financial concepts into the same vector space.

### 2. Vector Store & Indexing

* **Storage**: The vectors are stored locally using **LlamaIndex**'s default storage context.
* **Process**: Documents are chunked, embedded, and indexed. When a user asks a question, the system searches for the top most similar chunks in the database.

### 3. Generative Engine (LLM)

* **Model**: **Meta Llama 3.1** (via Ollama)
* **Role**: The LLM receives the user's question *plus* the retrieved context chunks. It acts as a reasoning engine to synthesize the answer based *strictly* on the provided context.

### 4. Hardware Acceleration

* **Optimization**: The system is optimized for Apple Silicon (M-series chips) using `MPS` (Metal Performance Shaders) for PyTorch and Ollama, ensuring low-latency inference.

---

## Technologies Used

* **Python 3.10+**
* `llama-index` - The Data Framework for LLMs (RAG orchestration)
* `streamlit` - Interactive Web Interface
* `ollama` - Local LLM Runner (Llama 3.1)
* `huggingface` - Embedding models
* `pypdf` - PDF text extraction
* `pytorch` - Tensor computations

---

## Project Structure

```text
irfa_quant_bot/
│
├── data/                     # Drop your documents here
├── storage/                  # Generated Vector Index (do not edit manually)
├── app.py                    # Main Dashboard Application (Streamlit)
├── ingest_data.py            # ETL Script: Reads data/ -> creates storage/
└── README.md                 
```

---

## Installation and Usage

### Prerequisites

```bash
Python 3.10 or higher
Ollama (Application installed and running)
```

### Installation

1. **Clone the repository and Setup Environment**

```bash
git clone https://github.com/elyas-elyas/Quant_AI_Bot
cd irfa_quant_bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Setup Ollama**

Ensure the Ollama app is running, then pull the model:

```bash
ollama pull llama3.1
```

3. **Ingest Your Data**

Place your PDF files in the `data/` folder, then run:

```bash
python ingest_data.py
```

*This will create the `storage/` folder containing the vector index.*

4. **Launch the Assistant**

```bash
# We use the python module syntax to ensure correct path resolution
./venv/bin/python -m streamlit run app.py
```

---

## Strengths & Limitations

### Strengths

* **Hallucination Control**: By restricting the LLM to the provided context, the risk of inventing false financial formulas is significantly reduced.
* **Auditability**: The source citation feature allows students/researchers to verify the information in the original course material immediately.
* **Cost & Privacy**: Zero API costs (unlike OpenAI) and complete data sovereignty.

### Limitations

* **OCR Quality**: The system relies on the text quality of the PDFs. Scanned images without OCR cannot be read.
* **Hardware Dependent**: Performance (speed of generation) depends on the local machine's RAM and GPU/NPU.
* **Static Knowledge**: If a new PDF is added, `ingest_data.py` must be re-run to update the index.

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **RAG** | Retrieval-Augmented Generation. A technique to optimize LLM output, referencing an authoritative knowledge base outside its training data. |
| **Embeddings** | Numerical representations of text where similar meanings are close in vector space. |
| **Top-k Retrieval** | The process of finding the 'k' most relevant document chunks to the user's query before sending them to the LLM. |
| **Context Window** | The limit on how much text (user query + retrieved documents) the LLM can process at once. |
| **Quantization** | Reducing the precision of the model (e.g., 4-bit) to run on consumer hardware with minimal loss of intelligence. |

---

## Resources and References

* **LlamaIndex Documentation**: [https://docs.llamaindex.ai/](https://docs.llamaindex.ai/)
* **Ollama**: [https://ollama.com/](https://ollama.com/)
* **Meta Llama 3**: [https://llama.meta.com/](https://llama.meta.com/)
* **Vaswani et al. (2017)**. "Attention Is All You Need". (The paper introducing the Transformer architecture used in this project).