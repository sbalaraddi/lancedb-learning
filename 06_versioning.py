"""
LanceDB Tutorial - Part 6: Versioning & Data Management
========================================================
Learn: Data versioning, updates, deletes, time-travel, compaction
"""

import lancedb
import pandas as pd
from datetime import datetime
import time

print("=" * 60)
print("PART 6: VERSIONING & DATA MANAGEMENT")
print("=" * 60)

# Connect to database
db = lancedb.connect("./my_database")

# LESSON 1: Understanding Versions
print("\n--- Lesson 1: Version Basics ---")

print("""
LANCEDB VERSIONING:
- Every write operation creates a new version
- Versions are immutable (cannot be modified)
- Old versions remain accessible (time-travel)
- Enables audit trails and rollback
- Automatic - no configuration needed
""")

# Create initial table
data_v1 = [
    {"id": 1, "name": "Alice", "score": 85, "vector": [0.1, 0.2, 0.3]},
    {"id": 2, "name": "Bob", "score": 90, "vector": [0.4, 0.5, 0.6]},
    {"id": 3, "name": "Charlie", "score": 75, "vector": [0.7, 0.8, 0.9]},
]

table = db.create_table("users_versioned", data=data_v1, mode="overwrite")
version_1 = table.version
print(f"✓ Created table, current version: {version_1}")
print(f"Record count: {table.count_rows()}")

# LESSON 2: Adding Data Creates New Version
print("\n--- Lesson 2: Versions from Data Modifications ---")

time.sleep(0.5)  # Small delay to see timestamp difference

# Add more data
data_v2 = [
    {"id": 4, "name": "David", "score": 88, "vector": [0.11, 0.22, 0.33]},
    {"id": 5, "name": "Eve", "score": 92, "vector": [0.44, 0.55, 0.66]},
]
table.add(data_v2)
version_2 = table.version
print(f"\n✓ Added 2 records, new version: {version_2}")
print(f"Record count: {table.count_rows()}")

# LESSON 3: Updates Create New Versions
print("\n--- Lesson 3: Updating Data ---")

time.sleep(0.5)

# Update a record
print("\nUpdating Bob's score...")
table.update(where="name = 'Bob'", values={"score": 95})
version_3 = table.version
print(f"✓ Updated record, new version: {version_3}")

# Verify update
bob_record = table.search().where("name = 'Bob'").to_pandas()
print(f"Bob's new score: {bob_record.iloc[0]['score']}")

# LESSON 4: Deletes Create New Versions
print("\n--- Lesson 4: Deleting Data ---")

time.sleep(0.5)

# Delete a record
print("\nDeleting Charlie...")
table.delete("name = 'Charlie'")
version_4 = table.version
print(f"✓ Deleted record, new version: {version_4}")
print(f"Record count: {table.count_rows()}")

# LESSON 5: Viewing Version History
print("\n--- Lesson 5: Version History ---")

# List all versions
print("\nVersion history:")
versions = table.list_versions()
print(f"\nTotal versions: {len(versions)}")

for v in versions[:5]:  # Show first 5
    print(f"Version {v['version']}: {v['timestamp']} - {v['metadata']}")

# LESSON 6: Time-Travel (Checkout Old Versions)
print("\n--- Lesson 6: Time-Travel to Previous Versions ---")

print(f"\nCurrent version: {table.version}")
current_data = table.search().to_pandas()
print(f"Current records: {len(current_data)}")
print(current_data[['id', 'name', 'score']])

# Go back to version 1 (original 3 records)
print(f"\n⏰ Traveling back to version {version_1}...")
table.checkout(version=version_1)
print(f"Now at version: {table.version}")

past_data = table.search().to_pandas()
print(f"Records in version {version_1}: {len(past_data)}")
print(past_data[['id', 'name', 'score']])

# Return to latest version
print("\n⏰ Returning to latest version...")
table.checkout_latest()
print(f"Back to version: {table.version}")
print(f"Record count: {table.count_rows()}")

# LESSON 7: Practical Use Cases for Versioning
print("\n--- Lesson 7: Versioning Use Cases ---")

print("""
USE CASES:

1. Audit Trail:
   - Track all data changes
   - Compliance and regulation
   - Security investigation

2. Rollback:
   - Undo bad updates
   - Recover from errors
   - A/B testing rollback

3. Time-Travel Queries:
   - "What did the data look like yesterday?"
   - Historical analysis
   - Reproduce old results

4. Data Recovery:
   - Restore accidentally deleted records
   - Recover from corruption
   - Compare versions

5. Experimentation:
   - Try changes without risk
   - Compare experiment results
   - Quick rollback if needed
""")

# LESSON 8: Version Management and Cleanup
print("\n--- Lesson 8: Version Cleanup ---")

print("""
WHY CLEANUP?
- Old versions consume disk space
- Each version stores changes
- Accumulates over time

CLEANUP STRATEGIES:
1. Keep recent versions only
2. Keep versions by age threshold
3. Keep specific important versions
""")

# Example: Cleanup old versions
print("\nCleaning up versions older than 0 days (for demo)...")
# In production, use older_than_days=7 or more
try:
    # Note: This might not delete all in demo due to timing
    stats = table.cleanup_old_versions(older_than_days=0, delete_unverified=False)
    print("✓ Cleanup completed")
except Exception as e:
    print(f"Note: Cleanup - {e}")

# LESSON 9: Compaction
print("\n--- Lesson 9: Data Compaction ---")

print("""
COMPACTION:
- Optimizes storage after many updates/deletes
- Merges small files into larger ones
- Improves read performance
- Reclaims disk space

WHEN TO COMPACT:
✅ After many updates/deletes
✅ Query performance degrades
✅ Many small files exist
✅ Regular maintenance (weekly/monthly)
""")

# Compact the table
print("\nCompacting table...")
try:
    stats = table.compact_files()
    print("✓ Compaction completed")
    print(f"Table optimized for better performance")
except Exception as e:
    print(f"Note: Compaction - {e}")

# LESSON 10: Best Practices for Data Management
print("\n--- Lesson 10: Data Management Best Practices ---")

print("""
BEST PRACTICES:

1. UPDATES:
   ✅ Batch updates when possible
   ✅ Use specific WHERE clauses
   ❌ Avoid updating entire table
   
   # Good
   table.update(where="category = 'A'", values={"status": "active"})
   
   # Bad - updates everything
   for row in all_rows:
       table.update(where=f"id = {row.id}", values=...)

2. DELETES:
   ✅ Soft delete with status flag (recommended)
   ✅ Hard delete periodically
   
   # Soft delete (preferred)
   table.update(where="id = 5", values={"deleted": True})
   query: .where("deleted = False")
   
   # Hard delete
   table.delete("id = 5")

3. VERSIONING:
   ✅ Regular cleanup (weekly/monthly)
   ✅ Keep 7-30 days of versions
   ✅ Archive critical versions separately
   
   # Regular cleanup
   table.cleanup_old_versions(older_than_days=7)

4. COMPACTION:
   ✅ Compact after batch operations
   ✅ Schedule during low-traffic periods
   ✅ Run weekly/monthly based on update frequency
   
   # After bulk operations
   table.add(large_batch)
   table.compact_files()

5. BULK OPERATIONS:
   ✅ Add all data first, then index
   ✅ Batch operations for efficiency
   
   # Good workflow
   table.add(all_data)
   table.create_index()
   
   # Bad - many small additions
   for item in data:
       table.add([item])  # Slow!

6. MONITORING:
   - Track table size
   - Monitor query performance
   - Count versions regularly
   - Check disk usage
""")

# LESSON 11: Practical Example - Incremental Updates
print("\n--- Lesson 11: Real-World Update Pattern ---")

# Create a products table
products = [
    {"id": 1, "name": "Product A", "price": 100, "stock": 50, "vector": [0.1]*8},
    {"id": 2, "name": "Product B", "price": 200, "stock": 30, "vector": [0.2]*8},
    {"id": 3, "name": "Product C", "price": 150, "stock": 40, "vector": [0.3]*8},
]

product_table = db.create_table("products_managed", data=products, mode="overwrite")
print("✓ Created products table")

# Scenario: Daily inventory update
print("\n--- Daily Inventory Update Workflow ---")

# 1. Update stock levels
print("\n1. Updating stock after sales...")
product_table.update(where="id = 1", values={"stock": 45})
product_table.update(where="id = 2", values={"stock": 28})
print("✓ Stock updated")

# 2. Add new products
print("\n2. Adding new products...")
new_products = [
    {"id": 4, "name": "Product D", "price": 175, "stock": 60, "vector": [0.4]*8},
]
product_table.add(new_products)
print("✓ New products added")

# 3. Price adjustment
print("\n3. Running promotion - 10% off Product A...")
current = product_table.search().where("id = 1").to_pandas()
old_price = current.iloc[0]['price']
new_price = old_price * 0.9
product_table.update(where="id = 1", values={"price": new_price})
print(f"✓ Price updated: ${old_price} → ${new_price}")

# 4. Check current state
print("\n4. Current inventory:")
inventory = product_table.search().to_pandas()
print(inventory[['id', 'name', 'price', 'stock']])

# 5. If mistake, can rollback
print("\n5. Can rollback to any previous version if needed")
print(f"Current version: {product_table.version}")
print("Use: table.checkout(version=X) to rollback")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Every write operation creates a new version automatically
2. Versions are immutable - can safely time-travel
3. Use checkout(version=N) to go to specific version
4. Use checkout_latest() to return to current

5. Data Operations:
   - Add: table.add(data)
   - Update: table.update(where=..., values=...)
   - Delete: table.delete(where=...)
   - All create new versions

6. Maintenance:
   - Cleanup: table.cleanup_old_versions(older_than_days=7)
   - Compact: table.compact_files()
   - Monitor: table.count_rows(), table.version

7. Best Practices:
   - Batch operations when possible
   - Prefer soft deletes
   - Regular cleanup and compaction
   - Keep 7-30 days of versions

8. Use Cases:
   - Audit trail and compliance
   - Rollback from errors
   - Historical analysis
   - A/B testing

9. Versioning is FREE:
   - No performance overhead
   - Minimal storage cost
   - Automatic - no configuration

10. Production Pattern:
    - Daily: Updates and additions
    - Weekly: Version cleanup
    - Monthly: Compaction
    - As needed: Rollback
""")

print("\n✅ Versioning and data management tutorial completed!")
print("\nNext: Run 07_rag_system.py to build a complete RAG application")
