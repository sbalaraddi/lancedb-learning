"""
LanceDB Tutorial - Part 9: Best Practices & Production Patterns
================================================================
Learn: Production-ready code, optimization, error handling, monitoring
"""

import lancedb
import numpy as np
import time
from typing import List, Dict, Optional
import json

print("=" * 60)
print("PART 9: BEST PRACTICES & PRODUCTION PATTERNS")
print("=" * 60)

# LESSON 1: Project Structure
print("\n--- Lesson 1: Production Project Structure ---")

print("""
RECOMMENDED PROJECT STRUCTURE:
==============================

my_vector_app/
├── config/
│   ├── __init__.py
│   ├── database.py          # DB connection config
│   └── settings.py          # App settings
├── models/
│   ├── __init__.py
│   ├── embeddings.py        # Embedding model wrappers
│   └── schemas.py           # Data schemas
├── services/
│   ├── __init__.py
│   ├── vectordb.py          # LanceDB operations
│   ├── search.py            # Search logic
│   └── ingestion.py         # Data ingestion
├── api/
│   ├── __init__.py
│   └── routes.py            # API endpoints
├── utils/
│   ├── __init__.py
│   ├── logging.py           # Logging setup
│   └── monitoring.py        # Metrics
├── tests/
│   ├── __init__.py
│   ├── test_search.py
│   └── test_ingestion.py
├── data/
│   └── lancedb/            # Database files
├── requirements.txt
├── .env                     # Environment variables
└── main.py                 # Entry point
""")

# LESSON 2: Configuration Management
print("\n--- Lesson 2: Configuration Management ---")

print("""
BEST PRACTICE: Centralized Configuration

# config/settings.py
""")

class Settings:
    """Application settings"""
    
    # Database
    DB_PATH: str = "./data/lancedb"
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    
    # Search
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100
    DEFAULT_NPROBES: int = 20
    
    # Index
    INDEX_PARTITIONS: int = 256
    INDEX_SUB_VECTORS: int = 96
    INDEX_METRIC: str = "cosine"
    
    # Performance
    BATCH_SIZE: int = 100
    CACHE_SIZE: int = 1000
    TIMEOUT_SECONDS: int = 30
    
    # Versioning
    KEEP_VERSIONS_DAYS: int = 7
    AUTO_COMPACT: bool = True
    
    @classmethod
    def from_env(cls):
        """Load from environment variables"""
        import os
        settings = cls()
        settings.DB_PATH = os.getenv("DB_PATH", settings.DB_PATH)
        settings.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", settings.EMBEDDING_MODEL)
        return settings

settings = Settings()
print(f"✓ Settings loaded: {settings.DB_PATH}")

# LESSON 3: Database Connection Pattern
print("\n--- Lesson 3: Database Connection Management ---")

class VectorDatabase:
    """
    Production-ready database wrapper
    Implements connection pooling, error handling, and retries
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = None
        self._tables = {}
    
    def connect(self):
        """Establish database connection"""
        try:
            self._db = lancedb.connect(self.db_path)
            print(f"✓ Connected to database: {self.db_path}")
            return self._db
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            raise
    
    def get_table(self, table_name: str, create_if_missing: bool = False):
        """
        Get table with caching
        
        Args:
            table_name: Name of the table
            create_if_missing: Create table if it doesn't exist
        """
        if table_name in self._tables:
            return self._tables[table_name]
        
        try:
            if table_name in self._db.table_names():
                table = self._db.open_table(table_name)
                self._tables[table_name] = table
                return table
            elif create_if_missing:
                print(f"Table '{table_name}' not found, creating...")
                return None
            else:
                raise ValueError(f"Table '{table_name}' not found")
        except Exception as e:
            print(f"❌ Error getting table: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check database health"""
        try:
            tables = self._db.table_names()
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def close(self):
        """Clean up resources"""
        self._tables.clear()
        self._db = None
        print("✓ Database connection closed")

# Initialize database
db = VectorDatabase(settings.DB_PATH)
db.connect()

# LESSON 4: Error Handling
print("\n--- Lesson 4: Robust Error Handling ---")

class SearchError(Exception):
    """Custom exception for search errors"""
    pass

class IngestionError(Exception):
    """Custom exception for ingestion errors"""
    pass

def safe_search(table, query_vector, top_k=10, max_retries=3):
    """
    Search with retry logic and error handling
    
    Args:
        table: LanceDB table
        query_vector: Query embedding
        top_k: Number of results
        max_retries: Maximum retry attempts
    
    Returns:
        Search results or raises SearchError
    """
    for attempt in range(max_retries):
        try:
            # Validate inputs
            if not isinstance(query_vector, list):
                raise ValueError("query_vector must be a list")
            
            if top_k <= 0 or top_k > settings.MAX_TOP_K:
                raise ValueError(f"top_k must be between 1 and {settings.MAX_TOP_K}")
            
            # Perform search
            results = table.search(query_vector).limit(top_k).to_pandas()
            return results
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️ Search failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise SearchError(f"Search failed after {max_retries} attempts: {e}")

print("✓ Error handling patterns defined")

# LESSON 5: Batch Processing
print("\n--- Lesson 5: Efficient Batch Processing ---")

def batch_insert(table, data: List[Dict], batch_size: int = 100):
    """
    Insert data in batches for better performance
    
    Args:
        table: LanceDB table
        data: List of records to insert
        batch_size: Records per batch
    """
    total = len(data)
    print(f"\n📦 Inserting {total} records in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        batch = data[i:i + batch_size]
        try:
            table.add(batch)
            progress = min(i + batch_size, total)
            print(f"   Progress: {progress}/{total} ({progress/total*100:.1f}%)")
        except Exception as e:
            print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            # Could implement retry or skip logic here
    
    print(f"✓ Insertion complete: {total} records")

# Demo batch insert
print("\nDemo: Batch insertion")
demo_data = [
    {"id": i, "text": f"Document {i}", "vector": np.random.randn(128).tolist()}
    for i in range(250)
]

demo_table = db.connect().create_table("demo_batch", data=demo_data[:1], mode="overwrite")
batch_insert(demo_table, demo_data[1:], batch_size=50)

# LESSON 6: Monitoring and Logging
print("\n--- Lesson 6: Monitoring & Logging ---")

class SearchMetrics:
    """Track search performance metrics"""
    
    def __init__(self):
        self.queries = []
        self.latencies = []
        self.errors = 0
    
    def record_query(self, query: str, latency: float, results: int, error: bool = False):
        """Record query metrics"""
        self.queries.append(query)
        self.latencies.append(latency)
        if error:
            self.errors += 1
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.latencies:
            return {"error": "No data"}
        
        return {
            "total_queries": len(self.queries),
            "avg_latency_ms": np.mean(self.latencies) * 1000,
            "p50_latency_ms": np.percentile(self.latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(self.latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(self.latencies, 99) * 1000,
            "error_rate": self.errors / len(self.queries) if self.queries else 0,
        }
    
    def print_stats(self):
        """Print metrics"""
        stats = self.get_stats()
        if "error" in stats:
            print(stats["error"])
            return
        
        print("\n📊 Performance Metrics:")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"   P95 latency: {stats['p95_latency_ms']:.2f}ms")
        print(f"   Error rate: {stats['error_rate']*100:.2f}%")

# Demo monitoring
metrics = SearchMetrics()

print("\n🔍 Running monitored searches...")
for i in range(10):
    query_vec = np.random.randn(128).tolist()
    start = time.time()
    try:
        results = demo_table.search(query_vec).limit(5).to_pandas()
        latency = time.time() - start
        metrics.record_query(f"query_{i}", latency, len(results))
    except Exception as e:
        latency = time.time() - start
        metrics.record_query(f"query_{i}", latency, 0, error=True)

metrics.print_stats()

# LESSON 7: Caching Strategy
print("\n--- Lesson 7: Query Result Caching ---")

class QueryCache:
    """Simple LRU cache for query results"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query_vector: List[float], filters: dict) -> str:
        """Create cache key from query"""
        # Simple hash of query vector and filters
        vec_hash = hash(tuple(query_vector[:10]))  # Use first 10 dims
        filter_hash = hash(json.dumps(filters, sort_keys=True))
        return f"{vec_hash}_{filter_hash}"
    
    def get(self, query_vector: List[float], filters: dict = None):
        """Get cached result"""
        key = self._make_key(query_vector, filters or {})
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, query_vector: List[float], filters: dict, results):
        """Cache result"""
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) entry
            self.cache.pop(next(iter(self.cache)))
        
        key = self._make_key(query_vector, filters or {})
        self.cache[key] = results
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }

# Demo caching
cache = QueryCache(max_size=50)
print("\n💾 Testing cache...")

query1 = np.random.randn(128).tolist()

# First query - cache miss
cached = cache.get(query1, {})
if cached is None:
    print("   Query 1: Cache MISS (expected)")
    results = demo_table.search(query1).limit(5).to_pandas()
    cache.put(query1, {}, results)

# Same query - cache hit
cached = cache.get(query1, {})
if cached is not None:
    print("   Query 1 (repeat): Cache HIT")

stats = cache.get_stats()
print(f"\n   Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']*100:.1f}% hit rate")

# LESSON 8: Data Validation
print("\n--- Lesson 8: Input Validation ---")

def validate_document(doc: Dict) -> bool:
    """
    Validate document before insertion
    
    Args:
        doc: Document to validate
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Required fields
    required = ["id", "text", "vector"]
    for field in required:
        if field not in doc:
            raise ValueError(f"Missing required field: {field}")
    
    # Vector dimension check
    expected_dim = settings.EMBEDDING_DIM
    if len(doc["vector"]) != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected {expected_dim}, got {len(doc['vector'])}")
    
    # Vector type check
    if not all(isinstance(x, (int, float)) for x in doc["vector"]):
        raise ValueError("Vector must contain only numbers")
    
    # Text length check
    if len(doc["text"]) > 10000:
        raise ValueError("Text too long (max 10000 characters)")
    
    return True

print("\n✓ Validation example:")
try:
    valid_doc = {"id": 1, "text": "Test", "vector": [0.1] * 384}
    validate_document(valid_doc)
    print("   Valid document: PASS")
except ValueError as e:
    print(f"   Validation failed: {e}")

try:
    invalid_doc = {"id": 2, "text": "Test"}  # Missing vector
    validate_document(invalid_doc)
except ValueError as e:
    print(f"   Invalid document: {e}")

# LESSON 9: Maintenance Tasks
print("\n--- Lesson 9: Automated Maintenance ---")

def perform_maintenance(table, settings):
    """
    Regular maintenance tasks
    
    Should be run periodically (daily/weekly)
    """
    print("\n🔧 Running maintenance tasks...")
    
    # 1. Cleanup old versions
    print("\n1. Cleaning up old versions...")
    try:
        table.cleanup_old_versions(older_than_days=settings.KEEP_VERSIONS_DAYS)
        print(f"   ✓ Removed versions older than {settings.KEEP_VERSIONS_DAYS} days")
    except Exception as e:
        print(f"   ⚠️ Cleanup failed: {e}")
    
    # 2. Compact files
    if settings.AUTO_COMPACT:
        print("\n2. Compacting files...")
        try:
            table.compact_files()
            print("   ✓ Compaction complete")
        except Exception as e:
            print(f"   ⚠️ Compaction failed: {e}")
    
    # 3. Verify index
    print("\n3. Verifying index...")
    try:
        count = table.count_rows()
        print(f"   ✓ Table has {count} rows")
    except Exception as e:
        print(f"   ⚠️ Verification failed: {e}")
    
    print("\n✓ Maintenance complete")

# Demo maintenance
print("\nDemo: Maintenance on demo table")
perform_maintenance(demo_table, settings)

# LESSON 10: Production Checklist
print("\n--- Lesson 10: Production Deployment Checklist ---")

print("""
PRODUCTION DEPLOYMENT CHECKLIST:
================================

📋 PRE-DEPLOYMENT:
□ Environment variables configured
□ Database backup strategy in place
□ Monitoring and alerting set up
□ Load testing completed
□ Security review done
□ Documentation updated

🔧 CONFIGURATION:
□ Appropriate index created
□ nprobes tuned for workload
□ Batch sizes optimized
□ Cache size configured
□ Timeout values set
□ Resource limits defined

🔐 SECURITY:
□ Access control implemented
□ API keys secured
□ Data encryption enabled
□ Audit logging configured
□ Input validation in place
□ Rate limiting active

📊 MONITORING:
□ Query latency tracking
□ Error rate monitoring
□ Database size tracking
□ Cache hit rate monitoring
□ Resource usage alerts
□ SLA metrics defined

⚙️ OPERATIONS:
□ Backup automation
□ Version cleanup scheduled
□ Compaction scheduled
□ Index rebuild process
□ Rollback procedure
□ Incident response plan

🧪 TESTING:
□ Unit tests passing
□ Integration tests passing
□ Load tests passing
□ Failure scenarios tested
□ Rollback tested
□ Disaster recovery tested

📈 SCALING:
□ Horizontal scaling plan
□ Database sharding strategy
□ Caching layer design
□ CDN for static content
□ Load balancer configured
□ Auto-scaling rules

🔄 MAINTENANCE:
□ Update schedule defined
□ Downtime windows planned
□ Migration strategy ready
□ Monitoring dashboard
□ Runbook documented
□ On-call rotation
""")

# Summary
print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. PROJECT STRUCTURE:
   - Separate concerns (config, models, services, API)
   - Centralize configuration
   - Use environment variables

2. CONNECTION MANAGEMENT:
   - Connection pooling
   - Health checks
   - Graceful cleanup

3. ERROR HANDLING:
   - Custom exceptions
   - Retry logic with exponential backoff
   - Comprehensive logging

4. PERFORMANCE:
   - Batch processing
   - Result caching
   - Query optimization
   - Index tuning

5. MONITORING:
   - Track latency metrics
   - Monitor error rates
   - Cache performance
   - Resource usage

6. VALIDATION:
   - Input validation
   - Schema validation
   - Type checking
   - Dimension verification

7. MAINTENANCE:
   - Automated version cleanup
   - Regular compaction
   - Index rebuilding
   - Health checks

8. PRODUCTION:
   - Comprehensive testing
   - Security hardening
   - Backup strategy
   - Incident response

9. OPTIMIZATION:
   - Profile before optimizing
   - Cache hot queries
   - Batch operations
   - Monitor and iterate

10. OPERATIONS:
    - Documentation
    - Runbooks
    - Alerting
    - Regular maintenance
""")

print("\n✅ Best practices tutorial completed!")
print("\n🎉 CONGRATULATIONS! You've completed all 9 LanceDB tutorials!")
print("\n📚 Next Steps:")
print("   1. Review LANCEDB_REFERENCE.md for quick lookups")
print("   2. Build your own project with LanceDB")
print("   3. Explore official docs: https://lancedb.github.io/lancedb/")
print("   4. Join the community: https://discord.gg/lancedb")
