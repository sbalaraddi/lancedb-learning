"""
LanceDB Tutorial - Part 7: Building a RAG System
=================================================
Learn: Build a complete Retrieval Augmented Generation pipeline
"""

import lancedb
import numpy as np

print("=" * 60)
print("PART 7: BUILDING A RAG (Retrieval Augmented Generation) SYSTEM")
print("=" * 60)

# LESSON 1: What is RAG?
print("\n--- Lesson 1: Understanding RAG ---")

print("""
RAG ARCHITECTURE:
=================

Traditional LLM Problem:
❌ Limited knowledge cutoff
❌ No access to private data
❌ Can hallucinate facts

RAG Solution:
✅ Retrieve relevant documents from knowledge base
✅ Augment LLM prompt with retrieved context
✅ Generate accurate, grounded responses

RAG Pipeline:
1. Document Ingestion → Chunk → Embed → Store in Vector DB
2. User Query → Embed query
3. Similarity Search → Retrieve top-K relevant chunks
4. Augment Prompt → Query + Retrieved context
5. Generate Response → LLM produces answer
""")

# LESSON 2: Document Ingestion
print("\n--- Lesson 2: Document Ingestion ---")

# Simulate a knowledge base
documents = [
    {
        "doc_id": "doc1",
        "title": "Company Policy - Remote Work",
        "content": "Employees are allowed to work remotely up to 3 days per week. Remote work requests must be approved by direct managers. All remote workers must be available during core hours 10 AM - 3 PM EST.",
        "category": "Policy",
        "source": "HR Handbook 2024"
    },
    {
        "doc_id": "doc2",
        "title": "Company Policy - Time Off",
        "content": "Full-time employees receive 15 days of paid time off annually. PTO requests should be submitted at least 2 weeks in advance. Unused PTO can be rolled over up to 5 days.",
        "category": "Policy",
        "source": "HR Handbook 2024"
    },
    {
        "doc_id": "doc3",
        "title": "Technical Guide - API Authentication",
        "content": "Our API uses OAuth 2.0 for authentication. Obtain an access token by sending client credentials to /auth/token endpoint. Tokens expire after 1 hour. Include the token in Authorization header.",
        "category": "Technical",
        "source": "API Documentation"
    },
    {
        "doc_id": "doc4",
        "title": "Technical Guide - Database Setup",
        "content": "Install PostgreSQL 14 or higher. Create a database named 'app_db'. Run migrations using 'npm run migrate'. Default port is 5432. Configure connection in .env file.",
        "category": "Technical",
        "source": "Developer Guide"
    },
    {
        "doc_id": "doc5",
        "title": "Company Policy - Benefits",
        "content": "Health insurance coverage starts after 30 days of employment. Company matches 401k contributions up to 5%. Dental and vision insurance available. Annual health checkup reimbursed.",
        "category": "Policy",
        "source": "Benefits Guide"
    },
    {
        "doc_id": "doc6",
        "title": "Technical Guide - Deployment",
        "content": "Deploy to production using CI/CD pipeline. Push to main branch triggers automatic deployment. Staging environment available for testing. Use 'npm run deploy:staging' for manual staging deployment.",
        "category": "Technical",
        "source": "DevOps Guide"
    },
]

print(f"📚 Knowledge Base: {len(documents)} documents")
for doc in documents:
    print(f"  - {doc['title']} ({doc['category']})")

# LESSON 3: Creating Embeddings (Simulated)
print("\n--- Lesson 3: Creating Embeddings ---")

print("""
In production, use real embedding models:
- sentence-transformers (local, free)
- OpenAI embeddings (API, paid)
- Cohere embeddings (API, paid)

Example with sentence-transformers:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text)
```
""")

# Simulate embeddings (in real app, use actual embedding model)
embedding_dim = 384

def simulate_embedding(text, category):
    """Simulate embeddings based on content and category"""
    np.random.seed(hash(text) % (2**32))
    
    # Create different embeddings for different categories
    if category == "Policy":
        base = np.random.normal(0.5, 0.1, embedding_dim)
    else:  # Technical
        base = np.random.normal(-0.5, 0.1, embedding_dim)
    
    # Add some content-specific variation
    content_hash = hash(text[:50]) % 100 / 100.0
    noise = np.random.normal(content_hash, 0.05, embedding_dim)
    
    return (base + noise).tolist()

# Add embeddings to documents
print("\n🔄 Generating embeddings...")
for doc in documents:
    doc["embedding"] = simulate_embedding(doc["content"], doc["category"])

print("✓ Embeddings created")

# LESSON 4: Store in LanceDB
print("\n--- Lesson 4: Storing in Vector Database ---")

db = lancedb.connect("./my_database")
kb_table = db.create_table("knowledge_base", data=documents, mode="overwrite")
print(f"✓ Stored {kb_table.count_rows()} documents in LanceDB")

# Create index for faster retrieval
# Note: With only 6 documents, we skip indexing (indexes need 10K+ vectors to be useful)
# In production with large datasets, uncomment the index creation below:
# kb_table.create_index(
#     vector_column_name="embedding",
#     metric="cosine",
#     num_partitions=256,
#     num_sub_vectors=96
# )
print("✓ Data stored (index skipped - dataset too small)")

# LESSON 5: Query Processing
print("\n--- Lesson 5: Processing User Queries ---")

def retrieve_context(query_text, category_filter=None, top_k=3):
    """
    Retrieve relevant documents for a query
    
    Args:
        query_text: User's question
        category_filter: Optional category to filter by
        top_k: Number of documents to retrieve
    
    Returns:
        List of relevant documents
    """
    # Generate query embedding (simulate)
    query_embedding = simulate_embedding(query_text, category_filter or "Policy")
    
    # Build search query
    search = kb_table.search(query_embedding, vector_column_name="embedding").limit(top_k)
    
    # Add filter if specified
    if category_filter:
        search = search.where(f"category = '{category_filter}'")
    
    # Execute search
    results = search.to_pandas()
    
    return results

# LESSON 6: Example Queries
print("\n--- Lesson 6: RAG in Action ---")

# Query 1: Remote work policy
print("\n" + "="*60)
print("QUERY 1: 'Can I work from home?'")
print("="*60)

query1 = "Can I work from home?"
results1 = retrieve_context(query1, category_filter="Policy", top_k=2)

print("\n📄 Retrieved Documents:")
for idx, row in results1.iterrows():
    print(f"\n{idx+1}. {row['title']}")
    print(f"   Content: {row['content'][:100]}...")
    print(f"   Relevance Score: {1 - row['_distance']:.3f}")

print("\n🤖 Augmented Prompt for LLM:")
context = "\n\n".join([f"Document: {row['title']}\n{row['content']}" 
                        for _, row in results1.iterrows()])
prompt = f"""Based on the following company information, answer the question.

Context:
{context}

Question: {query1}

Answer:"""
print(prompt[:300] + "...")

print("\n💡 Expected Answer: 'Yes, employees can work remotely up to 3 days per week with manager approval.'")

# Query 2: API authentication
print("\n" + "="*60)
print("QUERY 2: 'How do I authenticate with the API?'")
print("="*60)

query2 = "How do I authenticate with the API?"
results2 = retrieve_context(query2, category_filter="Technical", top_k=2)

print("\n📄 Retrieved Documents:")
for idx, row in results2.iterrows():
    print(f"\n{idx+1}. {row['title']}")
    print(f"   Content: {row['content'][:100]}...")
    print(f"   Relevance Score: {1 - row['_distance']:.3f}")

# Query 3: General query without filter
print("\n" + "="*60)
print("QUERY 3: 'What benefits are available?' (no category filter)")
print("="*60)

query3 = "What benefits are available?"
results3 = retrieve_context(query3, category_filter=None, top_k=2)

print("\n📄 Retrieved Documents:")
for idx, row in results3.iterrows():
    print(f"\n{idx+1}. {row['title']} ({row['category']})")
    print(f"   Content: {row['content'][:100]}...")

# LESSON 7: Hybrid Search for Better Retrieval
print("\n--- Lesson 7: Hybrid Search (Vector + Keyword) ---")

def hybrid_retrieve(query_text, keywords=None, category=None, top_k=3):
    """
    Enhanced retrieval with keyword filtering
    """
    query_embedding = simulate_embedding(query_text, category or "Policy")
    
    # Start with vector search
    search = kb_table.search(query_embedding, vector_column_name="embedding")
    
    # Add filters
    filters = []
    if category:
        filters.append(f"category = '{category}'")
    if keywords:
        for keyword in keywords:
            filters.append(f"content LIKE '%{keyword}%'")
    
    if filters:
        where_clause = " AND ".join(filters)
        search = search.where(where_clause)
    
    return search.limit(top_k).to_pandas()

print("\nExample: Search for remote work with keyword filter")
hybrid_results = hybrid_retrieve(
    "working from home",
    keywords=["remote", "managers"],
    category="Policy",
    top_k=2
)
print(f"Found {len(hybrid_results)} documents with hybrid search")
for idx, row in hybrid_results.iterrows():
    print(f"  - {row['title']}")

# LESSON 8: Building the Complete RAG Pipeline
print("\n--- Lesson 8: Complete RAG Pipeline ---")

def rag_query(user_question, category=None, top_k=3):
    """
    Complete RAG pipeline
    
    1. Embed the question
    2. Retrieve relevant documents
    3. Create augmented prompt
    4. (In production: Send to LLM)
    """
    print(f"\n🔍 Question: {user_question}")
    
    # Step 1: Retrieve
    print(f"\n1️⃣ Retrieving top {top_k} relevant documents...")
    results = retrieve_context(user_question, category, top_k)
    
    # Step 2: Build context
    print(f"2️⃣ Building context from {len(results)} documents...")
    context_parts = []
    for idx, row in results.iterrows():
        context_parts.append(
            f"[Source: {row['source']} - {row['title']}]\n{row['content']}"
        )
    context = "\n\n".join(context_parts)
    
    # Step 3: Create prompt
    print("3️⃣ Creating augmented prompt...")
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {user_question}

Provide a clear, concise answer based only on the context above. If the answer is not in the context, say so.

Answer:"""
    
    print("4️⃣ Prompt ready for LLM")
    
    return {
        "question": user_question,
        "retrieved_docs": len(results),
        "context": context,
        "prompt": prompt,
        "sources": results['title'].tolist()
    }

# Example usage
print("\n" + "="*60)
print("COMPLETE RAG EXAMPLE")
print("="*60)

result = rag_query("How many days of PTO do employees get?", category="Policy", top_k=2)
print(f"\n✓ Retrieved {result['retrieved_docs']} documents")
print(f"✓ Sources: {', '.join(result['sources'])}")
print(f"\n📝 Prompt length: {len(result['prompt'])} characters")
print("\n--- Prompt Preview ---")
print(result['prompt'][:400] + "...")

# LESSON 9: RAG Best Practices
print("\n--- Lesson 9: RAG Best Practices ---")

print("""
RAG OPTIMIZATION:

1. CHUNKING STRATEGY:
   ✅ Chunk documents into 200-500 words
   ✅ Maintain context (overlap chunks)
   ✅ Keep metadata with each chunk
   
2. EMBEDDING QUALITY:
   ✅ Use domain-specific models if available
   ✅ Fine-tune on your data
   ✅ Normalize embeddings for cosine similarity
   
3. RETRIEVAL:
   ✅ Start with top-K = 3-5
   ✅ Use hybrid search (vector + keywords)
   ✅ Apply filters early
   ✅ Re-rank if needed
   
4. CONTEXT WINDOW:
   ✅ Respect LLM token limits
   ✅ Prioritize most relevant chunks
   ✅ Include source citations
   
5. PROMPT ENGINEERING:
   ✅ Clear instructions
   ✅ Structured format
   ✅ Ask for citations
   ✅ Handle "not found" cases
   
6. EVALUATION:
   ✅ Track retrieval accuracy
   ✅ Monitor answer quality
   ✅ A/B test configurations
   ✅ Collect user feedback
""")

# LESSON 10: Production Considerations
print("\n--- Lesson 10: Production RAG System ---")

print("""
PRODUCTION CHECKLIST:

1. DATA PIPELINE:
   □ Automated document ingestion
   □ Incremental updates
   □ Error handling and retries
   □ Data validation

2. VECTOR DATABASE:
   □ Regular index rebuilding
   □ Backup and recovery
   □ Monitoring and alerting
   □ Scaling strategy

3. EMBEDDING:
   □ Model versioning
   □ Caching for common queries
   □ Batch processing for efficiency
   □ Fallback for model failures

4. SEARCH:
   □ Query validation
   □ Rate limiting
   □ Timeout handling
   □ Result caching

5. LLM INTEGRATION:
   □ API key management
   □ Token limit handling
   □ Cost optimization
   □ Fallback models

6. MONITORING:
   □ Retrieval latency
   □ LLM latency
   □ Total response time
   □ Accuracy metrics
   □ Cost per query

7. SECURITY:
   □ Access control
   □ Data encryption
   □ PII handling
   □ Audit logging
""")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. RAG = Retrieval + Augmentation + Generation
2. Pipeline: Ingest → Embed → Store → Retrieve → Augment → Generate
3. LanceDB provides fast vector storage and retrieval

4. Implementation Steps:
   a. Chunk and embed documents
   b. Store in LanceDB with metadata
   c. Create vector index
   d. Embed user queries
   e. Retrieve top-K similar chunks
   f. Augment LLM prompt with context
   g. Generate response

5. Optimization:
   - Use hybrid search (vector + keywords)
   - Apply category filters
   - Tune top-K (usually 3-5)
   - Monitor and iterate

6. Production:
   - Automate ingestion pipeline
   - Monitor performance
   - Handle errors gracefully
   - Scale as needed

7. Real Embedding Models:
   - sentence-transformers (free, local)
   - OpenAI ada-002 (paid, API)
   - Cohere embed (paid, API)

8. RAG Use Cases:
   - Customer support bots
   - Internal knowledge base
   - Document Q&A
   - Code documentation search
   - Legal document analysis
""")

print("\n✅ RAG system tutorial completed!")
print("\nNext: Run 08_semantic_search.py for semantic search patterns")
