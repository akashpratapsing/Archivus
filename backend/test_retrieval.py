"""
Test script to verify document chunking, embedding, indexing, and retrieval quality.
Run with: python test_retrieval.py
"""

import sys
import time
from chunking import process_and_chunk_document
from retrieval import vector_store
from generation import rag_generator

SAMPLE_DOC_TEXT = """# Acme Cloud Security & Architecture Whitepaper

## Architecture Overview
Acme Cloud platform uses a zero-trust microservices architecture deployed across multi-region Kubernetes clusters.
All intra-service communication is encrypted with mutual TLS (mTLS) using SPIFFE/SPIRE identity tokens.
Traffic enters through Envoy ingress gateways with rate limiting and automated DDoS mitigation.

## Authentication & Authorization
User authentication is managed via OAuth 2.0 and OpenID Connect (OIDC) backed by Okta and Google Workspace.
Access tokens are short-lived JSON Web Tokens (JWTs) with an expiration time of 15 minutes.
Fine-grained authorization uses Attribute-Based Access Control (ABAC) enforced by Open Policy Agent (OPA).
Service accounts must rotate credentials every 30 days via HashiCorp Vault.

## Data Retention and Compliance
Customer data at rest is encrypted with AES-256-GCM using customer-managed encryption keys (CMEK) stored in GCP Cloud KMS.
Backups are performed every 4 hours with point-in-time recovery (PITR) retained for 35 days.
Deleted accounts enter a soft-delete grace period of 30 days before permanent cryptographic erasure.
Acme complies with SOC 2 Type II, ISO 27001, and HIPAA Security Rule requirements.

## Incident Response SLA
Security incidents are classified into 4 severity levels:
- SEV-0 (Critical breach or data exposure): Initial response within 15 minutes, customer notification within 2 hours.
- SEV-1 (High risk outage or privilege escalation): Initial response within 30 minutes.
- SEV-2 (Moderate issue without data compromise): Initial response within 4 hours.
- SEV-3 (Low risk or informational bug): Initial response within 24 hours.
"""

def run_retrieval_test():
    print("=" * 60)
    print("🧪 Running RAG Retrieval Quality Test")
    print("=" * 60)

    filename = "Acme_Security_Whitepaper.md"
    content_bytes = SAMPLE_DOC_TEXT.encode("utf-8")

    print(f"\n1. Chunking test document '{filename}'...")
    chunks = process_and_chunk_document(filename, content_bytes, chunk_size=800, chunk_overlap=150)
    print(f"   Created {len(chunks)} chunks.")
    for i, c in enumerate(chunks):
        print(f"   - Chunk {i+1} [{c['metadata']['section']}]: {len(c['text'])} chars")

    print("\n2. Indexing chunks into Chroma...")
    try:
        vector_store.upsert_chunks(chunks)
        print(f"   Successfully indexed into Chroma. Total collection count: {vector_store.collection.count()}")
    except Exception as e:
        print(f"   ❌ Indexing failed (Check GEMINI_API_KEY): {e}")
        return

    test_queries = [
        ("What encryption is used for data at rest and how long are backups kept?", "Data Retention and Compliance"),
        ("What is the initial response SLA for a SEV-0 security incident?", "Incident Response SLA"),
        ("How often must service accounts rotate credentials?", "Authentication & Authorization"),
        ("What model of helicopter does the company fly?", "SHOULD NOT BE FOUND"),
    ]

    print("\n3. Testing Semantic Retrieval & Top-k Matches:")
    for query, expected_section in test_queries:
        print("-" * 50)
        print(f"🔍 Query: \"{query}\"")
        print(f"   Expected Section: {expected_section}")
        
        try:
            results = vector_store.search_similar_chunks(query, top_k=2)
            if not results:
                print("   ⚠️ No matching chunks found.")
                continue

            for idx, r in enumerate(results):
                sec = r['metadata'].get('section', 'Unknown')
                sim = r['similarity']
                snippet = r['text'][:120].replace('\n', ' ')
                print(f"   Match #{idx+1} [Sim: {sim:.3f}] [Section: {sec}]: \"{snippet}...\"")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

    print("\n✅ Retrieval test completed.")

if __name__ == "__main__":
    run_retrieval_test()
