"""
LanceDB Tutorial - Part 5: Full-Text Search & Hybrid Search
============================================================
Learn: FTS indexes, keyword search, combining with vector search
"""

import lancedb
import numpy as np

print("=" * 60)
print("PART 5: FULL-TEXT SEARCH & HYBRID SEARCH")
print("=" * 60)

# Connect to database
db = lancedb.connect("./my_database")

# LESSON 1: Creating Data for FTS
print("\n--- Lesson 1: Preparing Text Data ---")

# Create a document collection
documents = [
    {
        "id": 1,
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "author": "John Doe",
        "category": "AI",
        "tags": ["machine learning", "AI", "tutorial"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 2,
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
        "author": "Jane Smith",
        "category": "AI",
        "tags": ["deep learning", "neural networks", "AI"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 3,
        "title": "Python Programming Guide",
        "content": "Python is a versatile programming language widely used in web development, data science, and automation.",
        "author": "Bob Johnson",
        "category": "Programming",
        "tags": ["python", "programming", "tutorial"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 4,
        "title": "Data Science with Python",
        "content": "Data science combines statistics, machine learning, and programming to extract insights from data.",
        "author": "Alice Williams",
        "category": "Data Science",
        "tags": ["data science", "python", "machine learning"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 5,
        "title": "Neural Networks Explained",
        "content": "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "author": "John Doe",
        "category": "AI",
        "tags": ["neural networks", "AI", "deep learning"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 6,
        "title": "Web Development with JavaScript",
        "content": "JavaScript is essential for modern web development, enabling interactive and dynamic web applications.",
        "author": "Carol Davis",
        "category": "Programming",
        "tags": ["javascript", "web development", "programming"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 7,
        "title": "Statistical Analysis Basics",
        "content": "Statistical analysis involves collecting, analyzing, and interpreting data to discover patterns and trends.",
        "author": "David Brown",
        "category": "Statistics",
        "tags": ["statistics", "data analysis", "math"],
        "vector": np.random.randn(128).tolist()
    },
    {
        "id": 8,
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand, interpret, and generate human language using machine learning.",
        "author": "Jane Smith",
        "category": "AI",
        "tags": ["NLP", "machine learning", "AI"],
        "vector": np.random.randn(128).tolist()
    },
]

table = db.create_table("documents_fts", data=documents, mode="overwrite")
print(f"✓ Created table with {len(documents)} documents")

# LESSON 2: Creating Full-Text Search Index
print("\n--- Lesson 2: Creating FTS Index ---")

print("\nCreating FTS index on 'content' column...")
# Note: FTS syntax may vary by LanceDB version
# This creates a searchable index on text content
try:
    table.create_fts_index("content")
    print("✓ FTS index created on 'content' column")
except Exception as e:
    print(f"Note: FTS index creation - {e}")
    print("Continuing with basic text search...")

# LESSON 3: Keyword Search
print("\n--- Lesson 3: Keyword-Based Search ---")

# Search using SQL WHERE with LIKE
print("\n1. Search for 'machine learning' in content:")
ml_docs = table.search().where("content LIKE '%machine learning%'").to_pandas()
print(f"Found {len(ml_docs)} documents")
print(ml_docs[['title', 'author', 'category']])

print("\n2. Search for 'Python' in title or content:")
python_docs = table.search().where(
    "title LIKE '%Python%' OR content LIKE '%Python%'"
).to_pandas()
print(f"Found {len(python_docs)} Python-related documents")
print(python_docs[['title', 'category']])

# LESSON 4: Vector Search (Semantic)
print("\n--- Lesson 4: Semantic Vector Search ---")

# Create a query vector (simulating an embedding for "AI and machine learning")
query_vector = np.random.randn(128).tolist()

print("\n1. Semantic search for similar documents:")
semantic_results = table.search(query_vector).limit(3).to_pandas()
print("Top 3 semantically similar documents:")
print(semantic_results[['title', 'category', '_distance']])

# LESSON 5: Hybrid Search (Vector + Keyword)
print("\n--- Lesson 5: Hybrid Search (Best of Both Worlds) ---")

print("""
HYBRID SEARCH combines:
- Vector Search: Semantic similarity (meaning-based)
- Keyword Search: Exact term matching

Benefits:
✅ Find semantically similar content
✅ Ensure specific keywords are present
✅ Better precision and recall
""")

# Example 1: Semantic search with category filter
print("\n1. Similar AI documents (semantic + filter):")
hybrid_1 = (table.search(query_vector)
    .where("category = 'AI'")
    .limit(3)
    .to_pandas())
print(hybrid_1[['title', 'category', '_distance']])

# Example 2: Semantic search with keyword requirement
print("\n2. Similar docs that mention 'neural networks':")
hybrid_2 = (table.search(query_vector)
    .where("content LIKE '%neural networks%'")
    .limit(3)
    .to_pandas())
print(hybrid_2[['title', '_distance']])

# Example 3: Complex hybrid query
print("\n3. AI documents about 'machine learning' by John Doe:")
hybrid_3 = (table.search(query_vector)
    .where("category = 'AI' AND content LIKE '%machine learning%' AND author = 'John Doe'")
    .limit(3)
    .to_pandas())
print(f"Found {len(hybrid_3)} matching documents")
if len(hybrid_3) > 0:
    print(hybrid_3[['title', 'author']])

# LESSON 6: Search by Author and Tags
print("\n--- Lesson 6: Multi-Field Search ---")

# Search by author
print("\n1. Documents by 'Jane Smith':")
by_author = table.search().where("author = 'Jane Smith'").to_pandas()
print(by_author[['title', 'category']])

# Search by category
print("\n2. All AI category documents:")
ai_docs = table.search().where("category = 'AI'").to_pandas()
print(f"Found {len(ai_docs)} AI documents")
print(ai_docs[['title', 'author']])

# LESSON 7: Practical Hybrid Search Patterns
print("\n--- Lesson 7: Real-World Hybrid Search Patterns ---")

print("\nPattern 1: RAG with Keyword Filtering")
print("Use case: Answer questions using only recent documents mentioning specific terms")
print("""
# Pseudo-code
query_embedding = embed(user_question)
context_docs = table.search(query_embedding)
    .where("date > '2024-01-01' AND content LIKE '%important_term%'")
    .limit(5)
# Feed to LLM
""")

print("\nPattern 2: E-commerce Product Search")
print("Use case: Find similar products with specific attributes")
print("""
# Product similarity with filters
query_image_embedding = embed(user_upload)
similar_products = table.search(query_image_embedding)
    .where("category = 'shoes' AND price < 100 AND in_stock = true")
    .limit(20)
""")

print("\nPattern 3: Document Discovery")
print("Use case: Find related documents by topic and author")
print("""
# Topic-based with author preference
topic_embedding = embed("quantum computing")
results = table.search(topic_embedding)
    .where("author IN ('expert1', 'expert2') AND peer_reviewed = true")
    .limit(10)
""")

# LESSON 8: Search Result Ranking
print("\n--- Lesson 8: Understanding Search Rankings ---")

# Demonstrate distance-based ranking
query_vec = np.random.randn(128).tolist()
print("\n1. Vector search results are ranked by distance:")
ranked = table.search(query_vec).limit(5).to_pandas()
print(ranked[['title', '_distance']].to_string(index=False))
print("\nNote: Lower distance = more similar")

# LESSON 9: Combining Multiple Search Strategies
print("\n--- Lesson 9: Multi-Stage Search Pipeline ---")

print("""
ADVANCED SEARCH PIPELINE:

Stage 1: Broad Vector Search
  → Get top 100 semantically similar documents

Stage 2: Keyword Filtering  
  → Filter for required terms
  → Apply category/metadata filters

Stage 3: Re-ranking
  → Score by relevance
  → Boost recent documents
  → Consider user preferences

Stage 4: Final Selection
  → Return top K results
  → Diversify by category
""")

# Example implementation
print("\nExample: Two-stage search")
# Stage 1: Get candidates with vector search
candidates = table.search(query_vector).limit(20).to_pandas()
print(f"Stage 1: Found {len(candidates)} candidates via vector search")

# Stage 2: Filter candidates in Python (or use WHERE)
filtered = candidates[candidates['category'] == 'AI']
print(f"Stage 2: Filtered to {len(filtered)} AI documents")
print(filtered[['title', 'category', '_distance']].head())

# LESSON 10: Performance Considerations
print("\n--- Lesson 10: Hybrid Search Performance Tips ---")

print("""
OPTIMIZATION STRATEGIES:

1. Index Both Modalities:
   ✅ Create vector index for similarity search
   ✅ Create FTS index for keyword search
   
2. Filter Early:
   ✅ Apply filters in WHERE clause (pushed to DB)
   ❌ Don't filter in Python after fetching all data

3. Limit Results Appropriately:
   ✅ Use .limit(N) to control result size
   ✅ Fetch 2-3x what you need, then re-rank
   
4. Choose Right Metric:
   - Cosine: For normalized embeddings (most text)
   - L2: For absolute distance
   - Dot: For pre-scored vectors

5. Batch Queries:
   - If searching multiple queries, batch them
   - Reuse connections

6. Monitor Query Time:
   - Vector search: ~10-100ms (with index)
   - Filter: ~1-10ms
   - Total: Keep under 200ms for good UX
""")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Full-Text Search: Exact keyword matching
2. Vector Search: Semantic similarity matching
3. Hybrid Search: Combine both for best results

4. Use Cases:
   - FTS: "Find docs with exact term X"
   - Vector: "Find similar meaning"
   - Hybrid: "Find similar docs that mention X"

5. Implementation:
   - Use .where() for keyword filters
   - Use .search(vector) for semantic search
   - Combine them for hybrid search

6. Best Practices:
   - Index both text and vectors
   - Filter at database level
   - Rank by distance
   - Re-rank if needed

7. Typical Query Pattern:
   results = table.search(embedding)
       .where("keywords AND filters")
       .limit(K)
       .to_pandas()
""")

print("\n✅ Full-text and hybrid search tutorial completed!")
print("\nNext: Run 06_versioning.py to learn about data versioning")
