"""
LanceDB Tutorial - Part 3: Advanced Filtering & Queries
========================================================
Learn: SQL-like queries, complex filters, column selection
"""

import lancedb
import pandas as pd
from datetime import datetime, timedelta

print("=" * 60)
print("PART 3: ADVANCED FILTERING & QUERIES")
print("=" * 60)

# Connect to database
db = lancedb.connect("./my_database")

# Create a rich dataset for querying
print("\n--- Creating Sample E-commerce Dataset ---")

# Generate realistic product data
products = []
categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
statuses = ["in_stock", "out_of_stock", "discontinued"]

import random
random.seed(42)

for i in range(1, 51):
    category = random.choice(categories)
    products.append({
        "id": i,
        "name": f"Product {i}",
        "category": category,
        "price": round(random.uniform(10, 500), 2),
        "rating": round(random.uniform(1, 5), 1),
        "stock": random.randint(0, 100),
        "status": random.choice(statuses),
        "tags": random.sample(["new", "sale", "featured", "popular"], k=random.randint(0, 3)),
        "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
        "vector": [random.random() for _ in range(8)]
    })

table = db.create_table("products", data=products, mode="overwrite")
print(f"✓ Created products table with {len(products)} items")

# LESSON 1: Basic WHERE Clauses
print("\n--- Lesson 1: Basic Filtering ---")

# Equal condition
print("\n1. Filter by category:")
electronics = table.search().where("category = 'Electronics'").limit(5).to_pandas()
print(f"Found {len(electronics)} Electronics items")
print(electronics[['name', 'category', 'price']].head())

# Comparison operators
print("\n2. Price range filter:")
affordable = table.search().where("price < 50").limit(5).to_pandas()
print(f"Found {len(affordable)} items under $50")
print(affordable[['name', 'price', 'category']].head())

print("\n3. High-rated products:")
top_rated = table.search().where("rating >= 4.5").limit(5).to_pandas()
print(f"Found {len(top_rated)} products rated 4.5+")
print(top_rated[['name', 'rating', 'category']].head())

# LESSON 2: Compound Conditions
print("\n--- Lesson 2: Compound Conditions ---")

# AND operator
print("\n1. Cheap Electronics (AND condition):")
cheap_electronics = table.search().where(
    "category = 'Electronics' AND price < 100"
).limit(5).to_pandas()
print(cheap_electronics[['name', 'category', 'price']].head())

# OR operator
print("\n2. Books or Clothing (OR condition):")
books_or_clothing = table.search().where(
    "category = 'Books' OR category = 'Clothing'"
).limit(5).to_pandas()
print(books_or_clothing[['name', 'category', 'price']].head())

# Complex combinations
print("\n3. Complex filter:")
complex_filter = table.search().where(
    "(category = 'Electronics' OR category = 'Sports') AND price < 200 AND rating > 4.0"
).limit(5).to_pandas()
print(f"Found {len(complex_filter)} items matching complex criteria")
print(complex_filter[['name', 'category', 'price', 'rating']].head())

# LESSON 3: IN Operator
print("\n--- Lesson 3: IN Operator ---")

# Multiple category search
print("\n1. Search multiple categories:")
selected_categories = table.search().where(
    "category IN ('Electronics', 'Home', 'Sports')"
).limit(5).to_pandas()
print(selected_categories[['name', 'category']].head())

# Multiple status check
print("\n2. Available products:")
available = table.search().where(
    "status IN ('in_stock') AND stock > 10"
).limit(5).to_pandas()
print(f"Found {len(available)} available products")
print(available[['name', 'status', 'stock']].head())

# LESSON 4: String Matching (LIKE)
print("\n--- Lesson 4: String Pattern Matching ---")

# Contains pattern
print("\n1. Search product names (using comparison):")
# Note: LanceDB uses SQL-style LIKE
specific_products = table.search().where("name >= 'Product 1' AND name < 'Product 2'").limit(5).to_pandas()
print(specific_products[['name', 'category']].head())

# LESSON 5: NULL Checks
print("\n--- Lesson 5: NULL/NOT NULL Checks ---")

# Check for non-null values
print("\n1. Products with ratings:")
with_ratings = table.search().where("rating IS NOT NULL").limit(5).to_pandas()
print(f"Found {len(with_ratings)} products with ratings")

# LESSON 6: Column Selection
print("\n--- Lesson 6: Selecting Specific Columns ---")

# Select only needed columns for better performance
print("\n1. Select specific columns:")
selected_cols = table.search().select(["name", "price", "category"]).limit(5).to_pandas()
print(selected_cols)

print("\n2. Select for display:")
display_cols = table.search().where(
    "category = 'Electronics'"
).select(["id", "name", "price", "rating"]).limit(5).to_pandas()
print(display_cols)

# LESSON 7: Combining Vector Search with Filters
print("\n--- Lesson 7: Vector Search + Filters ---")

# Create a query vector
query_vector = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Vector search with category filter
print("\n1. Similar products in Electronics category:")
similar_electronics = (table.search(query_vector)
    .where("category = 'Electronics'")
    .limit(5)
    .to_pandas())
print(similar_electronics[['name', 'category', 'price', '_distance']].head())

# Vector search with price and rating filters
print("\n2. Similar affordable high-rated products:")
similar_quality = (table.search(query_vector)
    .where("price < 100 AND rating >= 4.0")
    .limit(5)
    .to_pandas())
print(similar_quality[['name', 'price', 'rating', '_distance']].head())

# LESSON 8: Aggregations (Count)
print("\n--- Lesson 8: Counting Results ---")

print("\n1. Total products:", table.count_rows())

# Count by applying filter
electronics_count = len(table.search().where("category = 'Electronics'").to_pandas())
print(f"2. Electronics count: {electronics_count}")

in_stock_count = len(table.search().where("status = 'in_stock'").to_pandas())
print(f"3. In-stock count: {in_stock_count}")

# LESSON 9: Practical Query Patterns
print("\n--- Lesson 9: Real-World Query Patterns ---")

# Pattern 1: Featured products
print("\n1. Featured Products Pattern:")
# Note: Array contains check might not be directly supported, use workarounds
featured = table.search().where(
    "status = 'in_stock' AND rating >= 4.0 AND price < 300"
).select(["name", "price", "rating", "category"]).limit(5).to_pandas()
print(featured)

# Pattern 2: Inventory check
print("\n2. Low Stock Alert Pattern:")
low_stock = table.search().where(
    "stock < 20 AND stock > 0 AND status = 'in_stock'"
).select(["name", "category", "stock"]).limit(5).to_pandas()
print(f"Low stock items: {len(low_stock)}")
print(low_stock)

# Pattern 3: Price range buckets
print("\n3. Price Range Analysis:")
budget = len(table.search().where("price < 50").to_pandas())
mid_range = len(table.search().where("price >= 50 AND price < 200").to_pandas())
premium = len(table.search().where("price >= 200").to_pandas())
print(f"Budget (<$50): {budget}")
print(f"Mid-range ($50-200): {mid_range}")
print(f"Premium (>$200): {premium}")

# LESSON 10: Performance Tips
print("\n" + "=" * 60)
print("QUERY OPTIMIZATION TIPS:")
print("=" * 60)
print("""
1. Use SELECT to fetch only needed columns
   ✅ Good: .select(['id', 'name', 'price'])
   ❌ Bad: Fetching all columns when you need just a few

2. Apply filters BEFORE vector search
   ✅ Good: .where("category = 'A'").search(vec)
   ❌ Bad: Search everything then filter in Python

3. Use indexes for large datasets (next lesson)
   
4. Combine filters efficiently:
   ✅ Good: "price < 100 AND category = 'Electronics'"
   ❌ Bad: Separate queries then merging results

5. Use IN for multiple value checks:
   ✅ Good: "category IN ('A', 'B', 'C')"
   ❌ Bad: "category = 'A' OR category = 'B' OR ..."

6. For counting, use count_rows() when possible
   ✅ Good: table.count_rows()
   ❌ Bad: len(table.search().to_pandas())

7. Limit results when testing:
   Always use .limit(N) during development
""")

print("\n✅ Advanced filtering tutorial completed!")
print("\nNext: Run 04_indexing.py to learn about performance optimization")
