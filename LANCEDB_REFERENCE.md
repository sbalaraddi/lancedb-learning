# LanceDB Complete Reference Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Core Concepts](#core-concepts)
4. [Database Operations](#database-operations)
5. [Table Operations](#table-operations)
6. [Vector Search](#vector-search)
7. [Filtering & Querying](#filtering--querying)
8. [Indexing](#indexing)
9. [Full-Text Search](#full-text-search)
10. [Data Management](#data-management)
11. [Best Practices](#best-practices)
12. [Common Patterns](#common-patterns)

---

## Introduction

**LanceDB** is an open-source, serverless vector database built on the Lance data format. It's designed for AI applications requiring efficient storage and retrieval of embeddings.

### Key Features
- ✅ **Serverless**: Embedded in your application, no separate server needed
- ✅ **Fast Vector Search**: Optimized ANN (Approximate Nearest Neighbor) search
- ✅ **Multi-Modal**: Store vectors with metadata, text, images
- ✅ **Disk-Based**: Handle datasets larger than RAM
- ✅ **SQL Support**: Familiar SQL-like query syntax
- ✅ **Versioning**: Built-in data versioning and time-travel
- ✅ **Zero Dependencies**: Pure Python, no C++ compilation required

---

## Installation & Setup

### Basic Installation
```bash
pip install lancedb
```

### With Optional Dependencies
```bash
# For embeddings generation
pip install lancedb sentence-transformers

# For OpenAI embeddings
pip install lancedb openai

# Full installation
pip install lancedb[all]
```

### Quick Start
```python
import lancedb

# Connect to database (creates if doesn't exist)
db = lancedb.connect("./my_database")

# Create table
data = [{"id": 1, "text": "hello", "vector": [0.1, 0.2]}]
table = db.create_table("my_table", data)
```

---

## Core Concepts

### 1. **Database**
- A directory on disk containing tables
- Connection is lightweight (no server process)
- Multiple databases can exist independently

### 2. **Table**
- Collection of records with a schema
- Must contain at least one vector column
- Stored in Apache Arrow format via Lance

### 3. **Vector Column**
- Fixed-dimension array of floats
- Used for similarity search
- Can have multiple vector columns per table

### 4. **Schema**
- Automatically inferred from first insert
- Can be explicitly defined using PyArrow
- Supports: int, float, string, binary, lists, structs

---

## Database Operations

### Connect to Database
```python
import lancedb

# Local database
db = lancedb.connect("./my_database")

# In-memory database (temporary)
db = lancedb.connect("memory://")

# Cloud storage (S3, GCS, Azure)
db = lancedb.connect("s3://my-bucket/my-database")
```

### List Tables
```python
# Get all table names
tables = db.table_names()
print(tables)  # ['products', 'users', 'documents']

# Check if table exists
if "products" in db.table_names():
    table = db.open_table("products")
```

### Drop Table
```python
db.drop_table("old_table")
```

---

## Table Operations

### Create Table

#### From List of Dicts
```python
data = [
    {"id": 1, "text": "hello", "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "text": "world", "vector": [0.4, 0.5, 0.6]},
]
table = db.create_table("my_table", data, mode="overwrite")
```

#### From Pandas DataFrame
```python
import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3],
    "text": ["a", "b", "c"],
    "vector": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
})
table = db.create_table("my_table", df)
```

#### From PyArrow Table
```python
import pyarrow as pa

schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 3))
])
table = db.create_table("my_table", schema=schema)
```

### Create Table Modes
```python
# Overwrite existing table
table = db.create_table("name", data, mode="overwrite")

# Create only if doesn't exist (default)
table = db.create_table("name", data, mode="create")

# Append to existing table
table = db.create_table("name", data, mode="append")
```

### Open Existing Table
```python
table = db.open_table("my_table")

# Or use dictionary access
table = db["my_table"]
```

### Add Data
```python
# Add more records
new_data = [{"id": 4, "text": "new", "vector": [0.7, 0.8, 0.9]}]
table.add(new_data)
```

### Count Records
```python
count = table.count_rows()
print(f"Total rows: {count}")
```

---

## Vector Search

### Basic Vector Search
```python
# Search for similar vectors
query_vector = [0.1, 0.2, 0.3]
results = table.search(query_vector).limit(5).to_pandas()
```

### Search Parameters

#### Limit Results
```python
results = table.search(query_vector).limit(10).to_pandas()
```

#### Distance Metrics
```python
# L2 (Euclidean) distance - default
results = table.search(query_vector).metric("L2").limit(5).to_pandas()

# Cosine similarity
results = table.search(query_vector).metric("cosine").limit(5).to_pandas()

# Dot product
results = table.search(query_vector).metric("dot").limit(5).to_pandas()
```

#### Select Specific Columns
```python
results = (table.search(query_vector)
    .select(["id", "text", "score"])
    .limit(5)
    .to_pandas())
```

#### Nprobes (Index Tuning)
```python
# Higher nprobes = more accurate but slower
# Lower nprobes = faster but less accurate
results = (table.search(query_vector)
    .nprobes(20)  # default is 20
    .limit(5)
    .to_pandas())
```

---

## Filtering & Querying

### WHERE Clause (SQL-style)
```python
# Basic filtering
results = table.search().where("price < 100").to_pandas()

# Multiple conditions with AND
results = table.search().where("category = 'electronics' AND price < 500").to_pandas()

# OR conditions
results = table.search().where("category = 'books' OR category = 'magazines'").to_pandas()

# IN operator
results = table.search().where("status IN ('active', 'pending')").to_pandas()

# LIKE operator
results = table.search().where("name LIKE '%apple%'").to_pandas()

# NULL checks
results = table.search().where("description IS NOT NULL").to_pandas()
```

### Combining Vector Search with Filters
```python
# Pre-filter: Filter first, then search
results = (table.search(query_vector)
    .where("category = 'electronics'")
    .limit(10)
    .to_pandas())

# This is efficient because filtering happens before vector search
```

### Select Columns
```python
# Select specific columns
results = table.search().select(["id", "name", "price"]).to_pandas()

# Select all columns (default)
results = table.search().to_pandas()
```

### Ordering
```python
# Order by a column
results = table.search().where("price > 0").order_by("price DESC").to_pandas()
```

---

## Indexing

### Why Index?
- **Speed**: 10-100x faster queries on large datasets
- **Trade-off**: Uses more disk space and memory
- **When**: Dataset > 100K vectors

### Create IVF-PQ Index
```python
# IVF-PQ: Inverted File Index with Product Quantization
# Most common index type, good balance of speed/accuracy/memory

table.create_index(
    metric="L2",           # or "cosine", "dot"
    num_partitions=256,    # More = better accuracy, more memory
    num_sub_vectors=96,    # PQ compression, should divide vector dim
)
```

### Index Parameters Explained

#### num_partitions (nlist)
- Number of clusters for IVF
- **Small dataset (< 1M)**: 256-1024
- **Medium (1M-10M)**: 1024-4096
- **Large (> 10M)**: 4096-16384
- Rule of thumb: `sqrt(num_vectors)`

#### num_sub_vectors (M)
- Product Quantization compression
- Must divide vector dimension evenly
- **Higher = more accurate but larger index**
- Example: 768-dim vector → 96 sub_vectors

#### nprobes (Query Time)
- Number of partitions to search
- Set during query, not index creation
- **Trade-off**: accuracy vs speed
- Typical: 10-50 for good balance

### Index Example
```python
# Dataset: 1M vectors, 384 dimensions
table.create_index(
    metric="cosine",
    num_partitions=1024,
    num_sub_vectors=96  # 384 / 96 = 4
)

# Query with tuned nprobes
results = table.search(query_vector).nprobes(30).limit(10).to_pandas()
```

### Check Index Status
```python
# List indexes
stats = table.stats()
print(stats)
```

---

## Full-Text Search

### Create FTS Index
```python
# Create full-text search index on text column
table.create_fts_index("text_column")

# Or multiple columns
table.create_fts_index(["title", "description"])
```

### Full-Text Search
```python
# Search for keywords
results = table.search("machine learning").limit(10).to_pandas()
```

### Hybrid Search (Vector + FTS)
```python
# Combine semantic search with keyword search
results = (table.search(query_vector)
    .where("text_column LIKE '%important keyword%'")
    .limit(10)
    .to_pandas())
```

---

## Data Management

### Update Records
```python
# Update with SQL-style syntax
table.update(
    where="id = 5",
    values={"price": 150.00, "status": "sold"}
)

# Bulk update
table.update(
    where="category = 'electronics'",
    values={"discount": 0.10}
)
```

### Delete Records
```python
# Delete specific records
table.delete("id = 10")

# Delete multiple
table.delete("category = 'expired'")

# Delete with condition
table.delete("created_at < '2023-01-01'")
```

### Versioning

LanceDB keeps versions of your data automatically.

```python
# Check current version
version = table.version
print(f"Current version: {version}")

# List all versions
versions = table.list_versions()
for v in versions:
    print(f"Version {v.version}: {v.timestamp}")

# Checkout (time-travel to) previous version
table.checkout(version=5)

# Restore latest
table.checkout_latest()
```

### Compaction

Over time, delete/update operations create fragments.

```python
# Compact table to optimize storage and performance
table.compact_files()

# Cleanup old versions
table.cleanup_old_versions(older_than_days=7)
```

---

## Best Practices

### 1. **Vector Dimensions**
- Use consistent dimensions across all records
- Common sizes: 384, 768, 1536 (from embedding models)
- Normalize vectors for cosine similarity

### 2. **Batch Operations**
```python
# ✅ Good: Batch insert
table.add(large_list_of_records)

# ❌ Bad: One at a time
for record in records:
    table.add([record])  # Slow!
```

### 3. **Indexing Strategy**
```python
# Create index AFTER bulk data loading
table.add(all_my_data)
table.create_index()  # Do this once
```

### 4. **Memory Management**
```python
# Use generators for large datasets
def data_generator():
    for batch in large_dataset:
        yield batch

for batch in data_generator():
    table.add(batch)
```

### 5. **Query Optimization**
```python
# ✅ Good: Filter before vector search
results = table.search(vec).where("category = 'A'").limit(10)

# ❌ Less efficient: Search then filter in Python
all_results = table.search(vec).limit(1000).to_pandas()
filtered = all_results[all_results['category'] == 'A'][:10]
```

### 6. **Schema Design**
```python
# Include metadata for filtering
data = {
    "id": 1,
    "vector": [0.1, 0.2, ...],
    "text": "original content",
    "category": "label",       # For filtering
    "timestamp": "2024-01-01", # For time-based queries
    "metadata": {"key": "val"} # Additional context
}
```

---

## Common Patterns

### Pattern 1: RAG (Retrieval Augmented Generation)
```python
import lancedb
from sentence_transformers import SentenceTransformer

# Setup
db = lancedb.connect("./rag_db")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Index documents
documents = ["doc1 text...", "doc2 text...", ...]
data = [
    {"id": i, "text": doc, "vector": model.encode(doc).tolist()}
    for i, doc in enumerate(documents)
]
table = db.create_table("documents", data)
table.create_index(num_partitions=256)

# Query
query = "What is machine learning?"
query_vec = model.encode(query).tolist()
results = table.search(query_vec).limit(5).to_pandas()

# Use results with LLM
context = "\n".join(results['text'].tolist())
# Pass context to your LLM...
```

### Pattern 2: Semantic Search
```python
# Search by text, automatically convert to vector
from lancedb.embeddings import get_registry

model = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")

# Embed on write
table = db.create_table("docs", data, embedding_functions=[
    model.create("text", "vector")
])

# Search with text (auto-embedded)
results = table.search("machine learning").limit(5).to_pandas()
```

### Pattern 3: Multi-Vector Search
```python
# Different vector types in same table
data = [
    {
        "id": 1,
        "text": "...",
        "text_vector": [0.1, 0.2, ...],    # Text embedding
        "image_vector": [0.5, 0.6, ...],   # Image embedding
    }
]

# Search specific vector column
results = table.search(query_vec, vector_column_name="text_vector").limit(5)
```

### Pattern 4: Incremental Updates
```python
# Regular data ingestion pipeline
def ingest_new_data(new_records):
    table = db.open_table("my_table")
    table.add(new_records)
    
    # Rebuild index if needed (costly, do periodically)
    if table.count_rows() % 10000 == 0:
        table.create_index(replace=True)
```

### Pattern 5: Filtered Similarity Search
```python
# Find similar items within a category
results = (table.search(query_vector)
    .where("category = 'electronics' AND price < 1000")
    .metric("cosine")
    .limit(10)
    .to_pandas())
```

---

## Quick Reference Cheat Sheet

### Database
```python
db = lancedb.connect("path")
db.table_names()
db.drop_table("name")
```

### Table Creation
```python
table = db.create_table("name", data, mode="overwrite")
table = db.open_table("name")
```

### Search
```python
table.search(vector).limit(10).to_pandas()
table.search(vector).metric("cosine").limit(5)
table.search(vector).where("price < 100").limit(10)
table.search(vector).nprobes(30).limit(10)
```

### Filtering
```python
table.search().where("category = 'A'").to_pandas()
table.search().select(["id", "name"]).to_pandas()
```

### Indexing
```python
table.create_index(num_partitions=256)
table.create_fts_index("text_column")
```

### Data Operations
```python
table.add(new_data)
table.update(where="id = 5", values={"key": "val"})
table.delete("id = 5")
table.count_rows()
```

### Versioning
```python
table.version
table.checkout(version=5)
table.cleanup_old_versions()
```

---

## Troubleshooting

### Issue: "Vector dimension mismatch"
**Solution**: Ensure all vectors have same dimension
```python
# Check vector dimensions
assert len(vector) == 384  # or your expected dimension
```

### Issue: "Slow queries on large dataset"
**Solution**: Create an index
```python
table.create_index(num_partitions=256)
```

### Issue: "Out of memory"
**Solution**: Use batch processing
```python
for batch in chunked_data:
    table.add(batch)
```

### Issue: "Can't find recent data"
**Solution**: Check if you're on old version
```python
table.checkout_latest()
```

---

## Additional Resources

- **Official Docs**: https://lancedb.github.io/lancedb/
- **GitHub**: https://github.com/lancedb/lancedb
- **Discord Community**: https://discord.gg/lancedb
- **Examples**: https://github.com/lancedb/lancedb/tree/main/examples

---

*Last Updated: December 2025*
