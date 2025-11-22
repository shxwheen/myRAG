# myRAG - Financial Document QA System

A Retrieval-Augmented Generation (RAG) system for question answering on financial documents, evaluated on the FinanceBench benchmark.

## Installation

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Create `.env` file with API keys
```bash
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

3. Add PDF documents to `data/test_files/finance-bench-pdfs/` from `https://github.com/patronus-ai/financebench/tree/main/pdfs`

## Quick Start

```bash
# Build vector database from PDFs (standard chunking)
python src/create_database.py

# OR: Build with element-based chunking (better table handling)
python src/create_database_element_based.py

# Interactive query interface
python src/retrieval.py

# Bulk testing on FinanceBench (full dataset)
python src/bulk_testing.py --dataset financebench

# Bulk testing on subset (24 questions, 10 companies)
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv --top-k 10
```

## Project Structure

```
myRAG/
├── src/                              # Core source code
│   ├── create_database.py            # Standard chunking (1000 chars)
│   ├── create_database_element_based.py  # Element-based chunking (2000 chars, table-aware)
│   ├── retrieval.py                  # Interactive queries
│   ├── bulk_testing.py               # Bulk evaluation with configurable features
│   └── metadata_utils.py             # Metadata extraction for filtering
├── dataset_adapters/                 # Dataset loading adapters
├── evaluation/                       # Evaluation metrics
├── tests/                            # Modular feature tests
│   ├── test_baseline.py              # Semantic search only
│   ├── test_hybrid_only.py           # BM25 + Semantic
│   ├── test_metadata_only.py         # Metadata filtering only
│   ├── test_reranking_only.py        # Cross-encoder reranking only
│   └── test_all_features.py          # All features enabled
├── data/
│   ├── test_files/finance-bench-pdfs/  # Input PDFs
│   └── question_sets/                  # Question subsets for testing
│       └── financebench_subset_questions.csv  # 24 questions, 10 companies
├── chroma/                           # Vector database (auto-created)
└── bulk_runs/                        # Test results (auto-created)
```

## Database Creation Options

### Standard Chunking (`src/create_database.py`)

Uses RecursiveCharacterTextSplitter:
- **Chunk size:** 1000 characters
- **Chunk overlap:** 200 characters
- **Embedding model:** OpenAI text-embedding-3-large

```bash
python src/create_database.py
```

### Element-Based Chunking (`src/create_database_element_based.py`)

Uses Unstructured.io for table-aware parsing:
- **Max chunk size:** 2000 characters
- **Soft max:** 1500 characters (before creating new chunk)
- **Combine threshold:** 1000 characters (merge small sections)
- **Table handling:** Tables kept as complete units
- **Embedding model:** OpenAI text-embedding-3-large

```bash
python src/create_database_element_based.py
```

## Retrieval Features

The bulk testing framework (`src/bulk_testing.py`) supports three configurable retrieval enhancements:

### 1. Hybrid Search (BM25 + Semantic)
Combines keyword-based BM25 retrieval with semantic embedding search using 50/50 weighting.

### 2. Metadata Filtering
Extracts company names and fiscal years from questions, then filters retrieved chunks to match. Falls back to unfiltered results if filtering removes too many chunks.

### 3. Cross-Encoder Reranking
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank retrieved chunks by relevance to the question.

### Configuration

Features are enabled by default in `BulkTestConfig`:
```python
use_hybrid_search: bool = True      # BM25 + Semantic
use_metadata_filtering: bool = True  # Filter by company/year
use_reranking: bool = True          # Cross-encoder reranking
```

## Bulk Testing

### Command-Line Options

```bash
python src/bulk_testing.py \
  --dataset financebench \           # Dataset name
  --top-k 10 \                       # Number of chunks to retrieve
  --temperature 0.0 \                # Generation randomness (0=deterministic)
  --max-tokens 512 \                 # Max response length
  --subset path/to/questions.csv     # Optional: test on subset of questions
```

### Subset Testing

For faster iteration, use the pre-built subset (24 questions from 10 companies):

```bash
python src/bulk_testing.py --subset data/question_sets/financebench_subset_questions.csv --top-k 10
```

**Subset companies:** Adobe, Apple, Microsoft, Activision Blizzard, Amazon, 3M, Costco, CVS, Block, AES

### Modular Feature Tests

Test individual retrieval features in isolation:

```bash
# Baseline (semantic search only)
python tests/test_baseline.py

# Individual features
python tests/test_hybrid_only.py
python tests/test_metadata_only.py
python tests/test_reranking_only.py

# All features combined
python tests/test_all_features.py
```

## How It Works

### 1. Database Creation

**Process:**
1. Load PDFs from `data/test_files/finance-bench-pdfs/`
2. Split into chunks (standard or element-based)
3. Embed chunks using OpenAI text-embedding-3-large
4. Store in ChromaDB at `chroma/`

### 2. Query Processing (`src/bulk_testing.py`)

For each question:
1. **Retrieve** initial chunks (3x top-k if using filtering/reranking)
2. **Filter** by metadata (company, year) if enabled
3. **Rerank** with cross-encoder if enabled
4. **Generate** answer using Llama 3.1 70B via Together API
5. **Evaluate** with semantic similarity against gold answer

### 3. Evaluation Metrics

- **Semantic Similarity:** Cosine similarity between predicted and gold answer embeddings (0.0-1.0)
- **Per-question-type breakdown:** Aggregated stats by question category
- **Timing:** Retrieval and generation latency

## Output Files

Results saved to `bulk_runs/` with format:
```
{timestamp}_{dataset}_{model}_k{top_k}_t{temp}.csv
{timestamp}_{dataset}_{model}_k{top_k}_t{temp}.json
```

**CSV columns:**
- question_id, question, gold_answer, predicted_answer
- semantic_similarity (0.0-1.0)
- retrieval_time_ms, generation_time_ms
- sources (retrieved document names)
- question_type, company, doc_name (metadata)
- error (if prediction failed)

**JSON contains:**
- Overall statistics (avg/min/max similarity)
- Per-question-type breakdown
- Configuration used

## Models Used

| Component | Model |
|-----------|-------|
| Embeddings | OpenAI text-embedding-3-large |
| LLM | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo (Together API) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |

## Troubleshooting

**Database not found:**
- Run `python src/create_database.py` or `python src/create_database_element_based.py` first

**Module not found errors:**
- Run scripts from project root: `cd /path/to/myRAG`

**Rate limit errors:**
- System saves partial results automatically
- Check Together API / OpenAI quota

**Low similarity scores:**
- Verify PDFs match dataset expectations
- Try increasing top-k
- Check retrieved sources in CSV output
- Experiment with different feature combinations
