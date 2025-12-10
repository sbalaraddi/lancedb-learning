"""
LanceDB Tutorial - Part 2: Vector Search & Embeddings
======================================================
Learn: Creating embeddings, similarity search, distance metrics
"""

import lancedb
import numpy as np

print("=" * 60)
print("PART 2: VECTOR SEARCH & EMBEDDINGS")
print("=" * 60)

# Connect to database
db = lancedb.connect("./my_database")

# CONCEPT: What are Embeddings?
# Embeddings are numerical representations of data (text, images, etc.)
# Similar items have similar vectors (close in vector space)

print("\n--- Understanding Vector Similarity ---")

# Create sample data with meaningful vectors
# Let's represent fruits and vegetables with simple 3D vectors
data = [
    # Fruits (first dimension high, represents "sweetness")
    {"id": 1, "name": "Apple", "type": "Fruit", "vector": [0.9, 0.3, 0.1]},
    {"id": 2, "name": "Banana", "type": "Fruit", "vector": [0.95, 0.4, 0.15]},
    {"id": 3, "name": "Orange", "type": "Fruit", "vector": [0.85, 0.35, 0.12]},
    {"id": 4, "name": "Grape", "type": "Fruit", "vector": [0.92, 0.38, 0.13]},
    
    # Vegetables (first dimension low, second dimension high - "savory")
    {"id": 5, "name": "Carrot", "type": "Vegetable", "vector": [0.2, 0.8, 0.5]},
    {"id": 6, "name": "Broccoli", "type": "Vegetable", "vector": [0.15, 0.85, 0.6]},
    {"id": 7, "name": "Spinach", "type": "Vegetable", "vector": [0.1, 0.9, 0.7]},
    {"id": 8, "name": "Celery", "type": "Vegetable", "vector": [0.12, 0.88, 0.65]},
]

table = db.create_table("food_items", data=data, mode="overwrite")
print(f"✓ Created table with {len(data)} food items")

# LESSON 1: Basic Vector Search
print("\n--- Lesson 1: Find Similar Items ---")
# Query: Find items similar to "Apple-like" vector
query_vector = [0.9, 0.3, 0.1]  # Apple's vector
print(f"Query vector (Apple-like): {query_vector}")

results = table.search(query_vector).limit(3).to_pandas()
print("\nTop 3 most similar items:")
print(results[['name', 'type', '_distance']])
# Note: _distance column is automatically added (lower = more similar)

# LESSON 2: Distance Metrics
print("\n--- Lesson 2: Distance Metrics ---")

# L2 Distance (Euclidean) - DEFAULT
# Measures straight-line distance between vectors
print("\n1. L2 Distance (Euclidean):")
results_l2 = table.search(query_vector).metric("L2").limit(3).to_pandas()
print(results_l2[['name', '_distance']])

# Cosine Distance
# Measures angle between vectors (ignores magnitude)
# Good for text embeddings where direction matters more than magnitude
print("\n2. Cosine Distance:")
results_cosine = table.search(query_vector).metric("cosine").limit(3).to_pandas()
print(results_cosine[['name', '_distance']])

# Dot Product
# Measures projection of one vector onto another
print("\n3. Dot Product:")
results_dot = table.search(query_vector).metric("dot").limit(3).to_pandas()
print(results_dot[['name', '_distance']])

# LESSON 3: Realistic Embeddings with NumPy
print("\n--- Lesson 3: Working with Real Embeddings ---")

# Simulate more realistic embeddings (higher dimensions)
np.random.seed(42)

# Generate 128-dimensional embeddings
# In real world, you'd use models like sentence-transformers or OpenAI
documents = [
    {"id": 1, "text": "Machine learning is awesome", "category": "AI"},
    {"id": 2, "text": "Deep learning and neural networks", "category": "AI"},
    {"id": 3, "text": "Python programming tutorial", "category": "Programming"},
    {"id": 4, "text": "Data science with pandas", "category": "Data"},
    {"id": 5, "text": "Artificial intelligence applications", "category": "AI"},
    {"id": 6, "text": "Web development with JavaScript", "category": "Programming"},
]

# Generate random embeddings (in practice, use a real embedding model)
embedding_dim = 128
for doc in documents:
    # Create semi-realistic embeddings based on category
    if doc["category"] == "AI":
        base = np.random.normal(0.5, 0.1, embedding_dim)
    elif doc["category"] == "Programming":
        base = np.random.normal(-0.5, 0.1, embedding_dim)
    else:
        base = np.random.normal(0.0, 0.1, embedding_dim)
    
    doc["vector"] = base.tolist()

table_docs = db.create_table("documents", data=documents, mode="overwrite")
print(f"✓ Created documents table with {embedding_dim}D embeddings")

# Search for AI-related content
print("\n--- Searching for AI-related content ---")
# Create query vector similar to AI embeddings
query_ai = np.random.normal(0.5, 0.1, embedding_dim).tolist()
results = table_docs.search(query_ai).limit(3).to_pandas()
print(results[['text', 'category', '_distance']])

# LESSON 4: Combining Vector Search with Filters
print("\n--- Lesson 4: Filtered Vector Search ---")

# Search only within "AI" category
print("\nSearch for similar docs in 'AI' category only:")
results_filtered = (table_docs.search(query_ai)
                    .where("category = 'AI'")
                    .limit(3)
                    .to_pandas())
print(results_filtered[['text', 'category']])

# LESSON 5: Understanding Distance Scores
print("\n--- Lesson 5: Interpreting Distance Scores ---")

# Create a known similar and dissimilar item
reference_vec = [1.0, 0.0, 0.0]
similar_vec = [0.98, 0.02, 0.01]    # Very similar
dissimilar_vec = [0.0, 1.0, 0.0]    # Orthogonal

test_data = [
    {"id": 1, "name": "Reference", "vector": reference_vec},
    {"id": 2, "name": "Similar", "vector": similar_vec},
    {"id": 3, "name": "Different", "vector": dissimilar_vec},
]

test_table = db.create_table("distance_demo", data=test_data, mode="overwrite")

# Calculate distances
results = test_table.search(reference_vec).limit(3).to_pandas()
print("\nDistance from reference vector:")
print(results[['name', '_distance']])

# Calculate cosine similarity (1 - cosine distance = cosine similarity)
results_cos = test_table.search(reference_vec).metric("cosine").limit(3).to_pandas()
print("\nCosine distance from reference:")
print(results_cos[['name', '_distance']])
print("\nNote: Lower distance = more similar")
print("Cosine distance of 0 = identical direction")
print("Cosine distance of 2 = opposite direction")

# LESSON 6: Practical Tips
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Vectors represent items as points in high-dimensional space
2. Similar items have vectors close together (small distance)
3. Distance Metrics:
   - L2 (Euclidean): Straight-line distance, sensitive to magnitude
   - Cosine: Angle-based, good for text (ignores magnitude)
   - Dot: Projection-based, considers both angle and magnitude

4. Vector Search finds K nearest neighbors (KNN)
5. Combine vector search with filters for category-specific search
6. In real applications, use proper embedding models:
   - Text: sentence-transformers, OpenAI embeddings
   - Images: CLIP, ResNet embeddings
   - Audio: wav2vec, Whisper embeddings

7. Vector dimensions typically: 128, 384, 768, or 1536
8. Always normalize vectors when using cosine similarity
""")

print("\n✅ Vector search tutorial completed!")
print("\nNext: Run 03_filtering_queries.py to learn advanced querying")
