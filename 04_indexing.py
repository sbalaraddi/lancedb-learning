"""
LanceDB Tutorial - Part 4: Indexing for Performance
====================================================
Learn: Creating indexes, understanding ANN, performance tuning
"""

import lancedb
import numpy as np
import time

print("=" * 60)
print("PART 4: INDEXING FOR PERFORMANCE")
print("=" * 60)

# Connect to database
db = lancedb.connect("./my_database")

# LESSON 1: Why Indexing Matters
print("\n--- Lesson 1: Performance Without Index ---")

# Create a dataset large enough to see index benefits
print("Creating dataset with 10,000 vectors...")
np.random.seed(42)
vector_dim = 128

# Generate data
large_data = []
for i in range(10000):
    large_data.append({
        "id": i,
        "text": f"Document {i}",
        "category": f"Cat_{i % 10}",
        "vector": np.random.randn(vector_dim).tolist()
    })

table = db.create_table("large_dataset", data=large_data, mode="overwrite")
print(f"✓ Created table with {table.count_rows()} vectors")

# Benchmark without index
query_vector = np.random.randn(vector_dim).tolist()

print("\n--- Searching WITHOUT index ---")
start = time.time()
results = table.search(query_vector).limit(10).to_pandas()
time_no_index = time.time() - start
print(f"Search time: {time_no_index:.4f} seconds")
print("Top 3 results:")
print(results[['id', 'text', '_distance']].head(3))

# LESSON 2: Creating an Index
print("\n--- Lesson 2: Creating IVF-PQ Index ---")

print("\nCreating index...")
print("Index type: IVF-PQ (Inverted File Index with Product Quantization)")
print("This is the most common index type - good balance of speed/accuracy/memory")

start = time.time()
table.create_index(
    metric="L2",           # Distance metric: L2, cosine, or dot
    num_partitions=256,    # Number of clusters (more = better accuracy)
    num_sub_vectors=16,    # PQ compression (must divide vector_dim)
)
index_time = time.time() - start
print(f"✓ Index created in {index_time:.2f} seconds")

# LESSON 3: Querying with Index
print("\n--- Lesson 3: Searching WITH index ---")

start = time.time()
results_indexed = table.search(query_vector).limit(10).to_pandas()
time_with_index = time.time() - start
print(f"Search time: {time_with_index:.4f} seconds")
print("Top 3 results:")
print(results_indexed[['id', 'text', '_distance']].head(3))

# Show speedup
speedup = time_no_index / time_with_index
print(f"\n🚀 Speedup: {speedup:.2f}x faster with index!")

# LESSON 4: Understanding Index Parameters
print("\n--- Lesson 4: Index Parameters Explained ---")

print("""
IVF-PQ Index Parameters:
========================

1. num_partitions (nlist):
   - Number of clusters/cells in the inverted file
   - Higher = better accuracy but slower build time
   - Rules of thumb:
     * Small dataset (<1M): 256-1024
     * Medium (1M-10M): 1024-4096  
     * Large (>10M): 4096-16384
     * Formula: sqrt(num_vectors) is a good starting point

2. num_sub_vectors (M):
   - Product Quantization compression parameter
   - Must divide vector dimension evenly
   - Higher = more accurate but larger index
   - Example: 128-dim vector → use 8, 16, 32, 64, or 128
   
3. metric:
   - L2: Euclidean distance (default)
   - cosine: Cosine similarity
   - dot: Dot product
""")

# LESSON 5: Query-Time Tuning with nprobes
print("\n--- Lesson 5: Tuning nprobes ---")

print("""
nprobes: Number of partitions to search at query time
- Set during QUERY, not during index creation
- Higher nprobes = more accurate but slower
- Lower nprobes = faster but less accurate
- Typical range: 1-50
""")

# Test different nprobes values
nprobe_values = [1, 5, 10, 20, 50]

print("\nComparing nprobes values:")
print(f"{'nprobes':<10} {'Time (s)':<12} {'Distance to 1st result':<20}")
print("-" * 42)

for nprobe in nprobe_values:
    start = time.time()
    results = table.search(query_vector).nprobes(nprobe).limit(10).to_pandas()
    search_time = time.time() - start
    first_distance = results.iloc[0]['_distance']
    print(f"{nprobe:<10} {search_time:<12.6f} {first_distance:<20.6f}")

print("\nObservation: Higher nprobes → more accurate but slower")

# LESSON 6: Different Index Configurations
print("\n--- Lesson 6: Index Configuration Examples ---")

# Example 1: Small dataset (10K-100K vectors)
print("\n1. Small Dataset Configuration:")
print("""
table.create_index(
    metric="cosine",
    num_partitions=256,
    num_sub_vectors=16
)
# Use nprobes=10-20 for queries
""")

# Example 2: Medium dataset (100K-1M vectors)
print("\n2. Medium Dataset Configuration:")
print("""
table.create_index(
    metric="L2",
    num_partitions=1024,
    num_sub_vectors=32
)
# Use nprobes=20-30 for queries
""")

# Example 3: Large dataset (1M+ vectors)
print("\n3. Large Dataset Configuration:")
print("""
table.create_index(
    metric="cosine",
    num_partitions=4096,
    num_sub_vectors=64
)
# Use nprobes=30-50 for queries
""")

# LESSON 7: When to Create Index
print("\n--- Lesson 7: Index Best Practices ---")

print("""
WHEN TO CREATE INDEX:
✅ Dataset has > 10,000 vectors
✅ Queries are slow
✅ Read-heavy workload
✅ Data is mostly static

WHEN NOT TO INDEX:
❌ Dataset is small (< 10,000 vectors)
❌ Write-heavy workload (frequent updates)
❌ Exact search required (index uses approximation)

WORKFLOW:
1. Insert all data first
2. Create index once
3. Query with index
4. Rebuild index periodically if data changes significantly

Index Creation Timing:
- Small (10K): ~1 second
- Medium (100K): ~10 seconds  
- Large (1M): ~1-2 minutes
- Very Large (10M+): ~10-30 minutes
""")

# LESSON 8: Practical Example - E-commerce Search
print("\n--- Lesson 8: Practical Example ---")

# Create a product database with realistic size
print("\nCreating product database with 5000 items...")
products = []
categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]

for i in range(5000):
    products.append({
        "id": i,
        "name": f"Product {i}",
        "category": categories[i % len(categories)],
        "price": np.random.uniform(10, 500),
        "description_vector": np.random.randn(384).tolist(),  # Simulated embedding
    })

product_table = db.create_table("products_indexed", data=products, mode="overwrite")
print(f"✓ Created table with {len(products)} products")

# Create index
print("\nCreating optimized index for product search...")
product_table.create_index(
    vector_column_name="description_vector",  # Specify the vector column
    metric="cosine",  # Cosine for text embeddings
    num_partitions=512,  # sqrt(5000) ≈ 70, round up to power of 2
    num_sub_vectors=96,  # 384 / 96 = 4
)
print("✓ Index created")

# Search for similar products
query_desc = np.random.randn(384).tolist()
print("\nSearching for similar products...")

start = time.time()
similar_products = (product_table.search(query_desc, vector_column_name="description_vector")
    .where("category = 'Electronics'")  # Filter by category
    .nprobes(20)  # Tune for accuracy
    .limit(10)
    .to_pandas())
search_time = time.time() - start

print(f"Found {len(similar_products)} similar electronics in {search_time:.4f}s")
print(similar_products[['id', 'name', 'category', 'price']].head())

# LESSON 9: Monitoring Index Performance
print("\n--- Lesson 9: Index Statistics ---")

# Get table statistics
stats = product_table.stats()
print("\nTable Statistics:")
print(f"Total rows: {product_table.count_rows()}")
print("\nNote: Use Lance format tools for detailed index stats")

# LESSON 10: Index Maintenance
print("\n--- Lesson 10: Index Maintenance ---")

print("""
INDEX MAINTENANCE:

1. Adding Data:
   # Index is automatically updated
   table.add(new_data)
   # For large batch inserts, consider rebuilding index

2. Rebuilding Index:
   # After significant data changes
   table.create_index(..., replace=True)

3. When to Rebuild:
   - After inserting > 20% new data
   - Query performance degrades
   - Changing index parameters

4. Index Storage:
   - Stored in the same directory as data
   - Uses additional disk space (~10-30% of data size)
   - Automatically loaded on table open
""")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Indexes provide 10-100x speedup for large datasets
2. IVF-PQ is the default and most balanced index type
3. Create index AFTER bulk loading data
4. Tune nprobes at query time for speed/accuracy tradeoff
5. num_partitions ≈ sqrt(num_vectors) is a good starting point
6. num_sub_vectors must divide vector dimension
7. Rebuild index after significant data changes
8. Indexes use approximate search (ANN), not exact
9. For < 10K vectors, no index needed
10. Combine indexed vector search with filters for best results
""")

print("\n✅ Indexing tutorial completed!")
print("\nNext: Run 05_full_text_search.py to learn about FTS and hybrid search")
