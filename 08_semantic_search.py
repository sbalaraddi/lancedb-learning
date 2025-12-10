"""
LanceDB Tutorial - Part 8: Semantic Search Application
=======================================================
Learn: Build a production-ready semantic search system
"""

import lancedb
import numpy as np
import pandas as pd
import time
from datetime import datetime

print("=" * 60)
print("PART 8: SEMANTIC SEARCH APPLICATION")
print("=" * 60)

# LESSON 1: What is Semantic Search?
print("\n--- Lesson 1: Semantic vs Keyword Search ---")

print("""
KEYWORD SEARCH (Traditional):
- Matches exact words/phrases
- Example: "python programming" → finds docs with these exact words
- Misses: synonyms, related concepts
- Fast but limited

SEMANTIC SEARCH (Modern):
- Understands meaning and intent
- Example: "python programming" → also finds "Python coding", "Python development"
- Captures: synonyms, context, intent
- Requires embeddings

Comparison:
Query: "ML algorithms"

Keyword Search finds:
✓ "machine learning algorithms"
✗ "classification and regression methods"
✗ "neural network architectures"

Semantic Search finds:
✓ "machine learning algorithms"
✓ "classification and regression methods"
✓ "neural network architectures"
✓ "deep learning techniques"
""")

# LESSON 2: Building a Blog Search System
print("\n--- Lesson 2: Use Case - Blog Search ---")

# Create a blog post database
blog_posts = [
    {
        "post_id": 1,
        "title": "Getting Started with Machine Learning",
        "content": "Machine learning is a powerful approach to building intelligent systems. It involves training models on data to make predictions and decisions.",
        "author": "Alice Johnson",
        "category": "Tutorial",
        "tags": ["machine learning", "beginner", "AI"],
        "published_date": "2024-01-15",
        "views": 1500
    },
    {
        "post_id": 2,
        "title": "Deep Learning for Image Recognition",
        "content": "Convolutional neural networks have revolutionized computer vision. They can identify objects, faces, and scenes in images with high accuracy.",
        "author": "Bob Smith",
        "category": "Advanced",
        "tags": ["deep learning", "CNN", "computer vision"],
        "published_date": "2024-02-20",
        "views": 2200
    },
    {
        "post_id": 3,
        "title": "Natural Language Processing Basics",
        "content": "NLP enables computers to understand and generate human language. Applications include chatbots, translation, and sentiment analysis.",
        "author": "Carol White",
        "category": "Tutorial",
        "tags": ["NLP", "language", "text processing"],
        "published_date": "2024-03-10",
        "views": 1800
    },
    {
        "post_id": 4,
        "title": "Python Data Analysis with Pandas",
        "content": "Pandas is essential for data manipulation in Python. Learn to clean, transform, and analyze datasets efficiently with dataframes.",
        "author": "David Lee",
        "category": "Tutorial",
        "tags": ["python", "pandas", "data analysis"],
        "published_date": "2024-04-05",
        "views": 3000
    },
    {
        "post_id": 5,
        "title": "Web Development with FastAPI",
        "content": "FastAPI is a modern web framework for building APIs with Python. It's fast, easy to use, and supports async operations.",
        "author": "Eve Chen",
        "category": "Tutorial",
        "tags": ["python", "web", "API"],
        "published_date": "2024-05-12",
        "views": 2500
    },
    {
        "post_id": 6,
        "title": "Introduction to Neural Networks",
        "content": "Neural networks are the foundation of deep learning. They consist of layers of interconnected nodes that process information.",
        "author": "Alice Johnson",
        "category": "Tutorial",
        "tags": ["neural networks", "deep learning", "AI"],
        "published_date": "2024-06-18",
        "views": 1900
    },
    {
        "post_id": 7,
        "title": "Data Visualization with Matplotlib",
        "content": "Effective visualization helps communicate data insights. Matplotlib provides powerful tools for creating charts and graphs in Python.",
        "author": "Bob Smith",
        "category": "Tutorial",
        "tags": ["python", "visualization", "matplotlib"],
        "published_date": "2024-07-22",
        "views": 1600
    },
    {
        "post_id": 8,
        "title": "Reinforcement Learning Explained",
        "content": "Reinforcement learning trains agents to make decisions through trial and error. Applications include robotics and game AI.",
        "author": "Carol White",
        "category": "Advanced",
        "tags": ["reinforcement learning", "AI", "agents"],
        "published_date": "2024-08-30",
        "views": 1400
    },
]

print(f"\n📚 Blog Database: {len(blog_posts)} posts")

# LESSON 3: Creating Semantic Embeddings
print("\n--- Lesson 3: Generating Embeddings ---")

def create_blog_embedding(post):
    """
    Create embedding from title and content
    In production, use: sentence-transformers or OpenAI
    """
    # Combine title and content for embedding
    text = f"{post['title']}. {post['content']}"
    
    # Simulate embedding (use real model in production)
    np.random.seed(hash(text) % (2**32))
    
    # Create embeddings based on content themes
    embedding = np.random.randn(256)
    
    # Add theme-specific signals
    if any(word in text.lower() for word in ["machine learning", "deep learning", "neural"]):
        embedding[:64] += np.random.normal(1.0, 0.1, 64)
    if any(word in text.lower() for word in ["python", "pandas", "matplotlib"]):
        embedding[64:128] += np.random.normal(1.0, 0.1, 64)
    if any(word in text.lower() for word in ["web", "api", "fastapi"]):
        embedding[128:192] += np.random.normal(1.0, 0.1, 64)
    
    return embedding.tolist()

print("🔄 Generating embeddings for blog posts...")
for post in blog_posts:
    post["embedding"] = create_blog_embedding(post)
print("✓ Embeddings created")

# LESSON 4: Store and Index
print("\n--- Lesson 4: Storing in LanceDB ---")

db = lancedb.connect("./my_database")
blog_table = db.create_table("blog_posts", data=blog_posts, mode="overwrite")
print(f"✓ Stored {blog_table.count_rows()} blog posts")

# Create index
# Note: With only 8 documents, we skip indexing (need 10K+ vectors for meaningful index)
# In production with large datasets, create index like this:
# blog_table.create_index(
#     vector_column_name="embedding",
#     metric="cosine",
#     num_partitions=256,
#     num_sub_vectors=64
# )
print("✓ Data stored (index skipped - dataset too small)")

# LESSON 5: Semantic Search Function
print("\n--- Lesson 5: Building Search Interface ---")

class BlogSearch:
    """Semantic search engine for blog posts"""
    
    def __init__(self, table):
        self.table = table
    
    def search(self, query, category=None, top_k=5, min_views=None):
        """
        Semantic search with filters
        
        Args:
            query: Search query string
            category: Filter by category
            top_k: Number of results
            min_views: Minimum view count
        """
        # Create query embedding
        query_embedding = create_blog_embedding({"title": query, "content": query})
        
        # Build search
        search = self.table.search(query_embedding, vector_column_name="embedding").metric("cosine")
        
        # Apply filters
        filters = []
        if category:
            filters.append(f"category = '{category}'")
        if min_views:
            filters.append(f"views >= {min_views}")
        
        if filters:
            where_clause = " AND ".join(filters)
            search = search.where(where_clause)
        
        # Execute
        results = search.limit(top_k).to_pandas()
        
        # Add relevance score (1 - distance)
        results['relevance'] = 1 - results['_distance']
        
        return results
    
    def display_results(self, results):
        """Pretty print search results"""
        if len(results) == 0:
            print("No results found.")
            return
        
        print(f"\n📄 Found {len(results)} results:\n")
        for idx, row in results.iterrows():
            print(f"{idx + 1}. {row['title']}")
            print(f"   Author: {row['author']} | Category: {row['category']} | Views: {row['views']}")
            print(f"   Relevance: {row['relevance']:.3f}")
            print(f"   Preview: {row['content'][:100]}...")
            print()

# Create search engine
search_engine = BlogSearch(blog_table)

# LESSON 6: Example Searches
print("\n--- Lesson 6: Semantic Search Examples ---")

# Example 1: AI and ML
print("\n" + "="*60)
print("SEARCH 1: 'artificial intelligence and neural networks'")
print("="*60)
results1 = search_engine.search("artificial intelligence and neural networks", top_k=3)
search_engine.display_results(results1)

# Example 2: Python data tools
print("\n" + "="*60)
print("SEARCH 2: 'analyzing data with python libraries'")
print("="*60)
results2 = search_engine.search("analyzing data with python libraries", top_k=3)
search_engine.display_results(results2)

# Example 3: With category filter
print("\n" + "="*60)
print("SEARCH 3: 'learning algorithms' (Tutorial category only)")
print("="*60)
results3 = search_engine.search("learning algorithms", category="Tutorial", top_k=3)
search_engine.display_results(results3)

# Example 4: Popular posts
print("\n" + "="*60)
print("SEARCH 4: 'python programming' (min 2000 views)")
print("="*60)
results4 = search_engine.search("python programming", min_views=2000, top_k=3)
search_engine.display_results(results4)

# LESSON 7: Related Posts Feature
print("\n--- Lesson 7: 'Related Posts' Feature ---")

def find_related_posts(post_id, top_k=3):
    """Find posts similar to a given post"""
    # Get the post
    post = blog_table.search().where(f"post_id = {post_id}").to_pandas()
    
    if len(post) == 0:
        print(f"Post {post_id} not found")
        return
    
    post = post.iloc[0]
    print(f"\n📌 Original Post: {post['title']}")
    
    # Search for similar posts (excluding the original)
    embedding = post['embedding']
    results = blog_table.search(embedding, vector_column_name="embedding").limit(top_k + 1).to_pandas()
    
    # Remove the original post
    results = results[results['post_id'] != post_id][:top_k]
    results['relevance'] = 1 - results['_distance']
    
    print(f"\n🔗 Related Posts:")
    for idx, row in results.iterrows():
        print(f"  {idx + 1}. {row['title']} (Relevance: {row['relevance']:.3f})")

# Find posts related to "Deep Learning for Image Recognition"
find_related_posts(2, top_k=3)

# LESSON 8: Search Analytics
print("\n--- Lesson 8: Search Analytics ---")

def analyze_search_quality(query, expected_keywords):
    """Analyze search result quality"""
    results = search_engine.search(query, top_k=5)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"📊 Analysis:")
    print(f"   Results found: {len(results)}")
    
    if len(results) > 0:
        avg_relevance = results['relevance'].mean()
        print(f"   Average relevance: {avg_relevance:.3f}")
        print(f"   Top result relevance: {results.iloc[0]['relevance']:.3f}")
        
        # Check if expected keywords appear in top results
        top_titles = " ".join(results['title'].tolist()).lower()
        top_content = " ".join(results['content'].tolist()).lower()
        combined = top_titles + " " + top_content
        
        found_keywords = [kw for kw in expected_keywords if kw.lower() in combined]
        print(f"   Expected keywords found: {len(found_keywords)}/{len(expected_keywords)}")
        print(f"   Keywords: {', '.join(found_keywords)}")

# Analyze different queries
analyze_search_quality(
    "computer vision and image processing",
    ["image", "vision", "CNN", "deep learning"]
)

analyze_search_quality(
    "data manipulation and visualization",
    ["pandas", "matplotlib", "data", "visualization"]
)

# LESSON 9: Performance Benchmarking
print("\n--- Lesson 9: Performance Benchmarking ---")

def benchmark_search(num_queries=10):
    """Benchmark search performance"""
    test_queries = [
        "machine learning basics",
        "python programming",
        "data visualization",
        "neural networks",
        "web development",
    ]
    
    times = []
    
    print(f"\n⏱️ Running {num_queries} search queries...")
    for i in range(num_queries):
        query = test_queries[i % len(test_queries)]
        
        start = time.time()
        results = search_engine.search(query, top_k=5)
        elapsed = time.time() - start
        
        times.append(elapsed)
    
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Average search time: {avg_time*1000:.2f}ms")
    print(f"   P95 search time: {p95_time*1000:.2f}ms")
    print(f"   Queries per second: {1/avg_time:.1f}")
    print(f"   Total queries: {num_queries}")

benchmark_search(num_queries=20)

# LESSON 10: Production Features
print("\n--- Lesson 10: Production-Ready Features ---")

print("""
PRODUCTION SEMANTIC SEARCH SYSTEM:

1. SEARCH FEATURES:
   ✅ Fuzzy matching
   ✅ Autocomplete suggestions
   ✅ Search history
   ✅ Popular searches
   ✅ Personalization

2. FILTERING:
   ✅ Category/tag filters
   ✅ Date range filters
   ✅ Author filters
   ✅ Popularity filters
   ✅ Multi-filter support

3. RANKING:
   ✅ Semantic relevance
   ✅ Freshness boost
   ✅ Popularity boost
   ✅ Quality signals
   ✅ Personalization

4. PERFORMANCE:
   ✅ Response time < 100ms
   ✅ Caching popular queries
   ✅ Async search
   ✅ Result pagination

5. MONITORING:
   ✅ Query latency
   ✅ Result quality
   ✅ Click-through rate
   ✅ Search abandonment
   ✅ User feedback

6. IMPROVEMENTS:
   ✅ A/B testing
   ✅ Feedback loop
   ✅ Model updates
   ✅ Index optimization
   ✅ Query understanding
""")

# Example: Advanced Search with Ranking
print("\n--- Advanced Search with Custom Ranking ---")

def advanced_search(query, boost_recent=True, boost_popular=True, top_k=5):
    """Search with custom ranking"""
    # Base semantic search
    query_embedding = create_blog_embedding({"title": query, "content": query})
    results = blog_table.search(query_embedding, vector_column_name="embedding").limit(top_k * 2).to_pandas()
    
    # Calculate base relevance
    results['semantic_score'] = 1 - results['_distance']
    
    # Apply boosts
    results['final_score'] = results['semantic_score']
    
    if boost_popular:
        # Normalize views to 0-1 range
        max_views = results['views'].max()
        results['popularity_score'] = results['views'] / max_views
        results['final_score'] += results['popularity_score'] * 0.3
    
    if boost_recent:
        # Parse dates and calculate recency
        results['days_old'] = (datetime.now() - pd.to_datetime(results['published_date'])).dt.days
        max_days = results['days_old'].max()
        results['recency_score'] = 1 - (results['days_old'] / max_days)
        results['final_score'] += results['recency_score'] * 0.2
    
    # Sort by final score
    results = results.sort_values('final_score', ascending=False).head(top_k)
    
    return results

print("\n🎯 Search with ranking: 'python programming'")
advanced_results = advanced_search("python programming", top_k=3)
print("\nTop 3 Results (with boost):")
for idx, row in advanced_results.iterrows():
    print(f"{idx + 1}. {row['title']}")
    print(f"   Final Score: {row['final_score']:.3f} (Semantic: {row['semantic_score']:.3f}, Views: {row['views']})")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Semantic search understands meaning, not just keywords
2. Embeddings capture semantic similarity
3. LanceDB provides fast vector search infrastructure

4. Implementation:
   - Embed documents (title + content)
   - Store in LanceDB with metadata
   - Create vector index
   - Embed queries and search

5. Features to Implement:
   - Filtering (category, date, author)
   - Custom ranking (recency, popularity)
   - Related items
   - Search analytics

6. Performance:
   - Index for speed (<100ms queries)
   - Cache popular queries
   - Monitor and optimize

7. Production Considerations:
   - Real embedding models (sentence-transformers, OpenAI)
   - A/B testing
   - User feedback
   - Continuous improvement

8. Use Cases:
   - Blog/content search
   - E-commerce product discovery
   - Documentation search
   - Knowledge base search
   - Recommendation systems
""")

print("\n✅ Semantic search tutorial completed!")
print("\nNext: Run 09_best_practices.py for production patterns")
