"""
LanceDB Tutorial - Part 1: Basic Operations
============================================
Learn: Creating a database, inserting data, and querying
"""

import lancedb
import pandas as pd

# Step 1: Create/Connect to a LanceDB database
# LanceDB stores data in a directory on disk
db = lancedb.connect("./my_database")
print("✓ Connected to LanceDB database")

# Step 2: Create sample data
# LanceDB works with pandas DataFrames, lists of dicts, or PyArrow tables
data = [
    {"id": 1, "name": "Apple", "category": "Fruit", "price": 1.50, "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "name": "Banana", "category": "Fruit", "price": 0.80, "vector": [0.2, 0.3, 0.4]},
    {"id": 3, "name": "Carrot", "category": "Vegetable", "price": 0.60, "vector": [0.7, 0.8, 0.9]},
    {"id": 4, "name": "Broccoli", "category": "Vegetable", "price": 1.20, "vector": [0.6, 0.7, 0.8]},
    {"id": 5, "name": "Orange", "category": "Fruit", "price": 1.00, "vector": [0.15, 0.25, 0.35]},
]

# Step 3: Create a table
# If table exists, this will overwrite it (use mode="append" to add data)
table = db.create_table("products", data=data, mode="overwrite")
print(f"✓ Created table 'products' with {len(data)} rows")

# Step 4: Basic Queries
print("\n--- All Products ---")
results = table.search().limit(10).to_pandas()
print(results)

print("\n--- Filter by Category ---")
# Use SQL-style WHERE clause
fruits = table.search().where("category = 'Fruit'").to_pandas()
print(fruits)

print("\n--- Filter by Price Range ---")
affordable = table.search().where("price < 1.0").to_pandas()
print(affordable)

print("\n--- Complex Filtering ---")
# Combine multiple conditions
expensive_fruits = table.search().where("category = 'Fruit' AND price > 1.0").to_pandas()
print(expensive_fruits)

# Step 5: Vector Search (Similarity Search)
# Find items similar to a query vector
print("\n--- Vector Similarity Search ---")
query_vector = [0.1, 0.2, 0.3]  # Similar to Apple
similar_items = table.search(query_vector).limit(3).to_pandas()
print("Items similar to query vector [0.1, 0.2, 0.3]:")
print(similar_items[['name', 'category', 'price']])

# Step 6: Adding More Data
print("\n--- Adding More Data ---")
new_products = [
    {"id": 6, "name": "Tomato", "category": "Vegetable", "price": 0.90, "vector": [0.5, 0.6, 0.7]},
    {"id": 7, "name": "Grape", "category": "Fruit", "price": 2.00, "vector": [0.12, 0.22, 0.32]},
]
table.add(new_products)
print(f"✓ Added {len(new_products)} more products")

# Verify total count
total = table.search().limit(100).to_pandas()
print(total)
print(f"Total products in table: {len(total)}")

print("\n✅ Basic operations completed!")
print("\nKey Takeaways:")
print("1. LanceDB is serverless - just connect to a directory")
print("2. Data can be inserted as Python dicts, DataFrames, or PyArrow tables")
print("3. Use SQL WHERE clauses for filtering")
print("4. Vector search finds similar items automatically")
print("5. Tables are persistent on disk")
