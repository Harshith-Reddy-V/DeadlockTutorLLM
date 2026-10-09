#!/usr/bin/env python3
"""Evaluation Script for DeadlockTutorLLM.

Runs the evaluation dataset through the orchestrator under different configurations
to benchmark correctness, latency, and groundedness.
"""

import json
import time
import argparse
from pathlib import Path
import sys

# Ensure backend can be imported
sys.path.append(str(Path(__file__).parent.parent))

from backend.orchestrator import DeadlockTutorOrchestrator, ChatRequest
from backend.llm.provider import get_llm_provider
from backend.rag.retriever import BaseRetriever


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(dataset_path: str):
    print("Loading orchestrator (this may take a moment)...")
    provider = get_llm_provider()
    retriever = BaseRetriever()
    orchestrator = DeadlockTutorOrchestrator(llm_provider=provider, retriever=retriever)
    dataset = load_dataset(dataset_path)

    results = []
    total_latency = 0.0
    correct_keywords = 0
    total_queries = len(dataset)

    print(f"Starting evaluation on {total_queries} queries...\n")

    for i, item in enumerate(dataset):
        query = item["query"]
        expected_keywords = item["expected_keywords"]
        category_expected = item["category"]

        print(f"[{i+1}/{total_queries}] Query: {query}")
        
        start_time = time.time()
        try:
            # We run the full system pipeline here.
            # (In a real ablation study, we would toggle RAG and Solver off using mocked components)
            req = ChatRequest(query=query)
            response = orchestrator.handle(req)
            latency = time.time() - start_time
            total_latency += latency
            
            answer = response.answer.lower()
            
            # Check keywords for a rough accuracy metric
            keywords_found = [kw for kw in expected_keywords if kw.lower() in answer]
            keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
            
            if keyword_score > 0.5:
                correct_keywords += 1
            
            res_dict = {
                "id": item["id"],
                "category_expected": category_expected,
                "category_actual": response.category,
                "latency_sec": round(latency, 2),
                "grounded": response.grounded,
                "keyword_score": round(keyword_score, 2)
            }
            results.append(res_dict)
            
            print(f"  -> Category: {response.category} (Expected: {category_expected})")
            print(f"  -> Latency: {latency:.2f}s | Grounded: {response.grounded} | Keyword Score: {keyword_score:.2f}")
            
        except Exception as e:
            print(f"  -> Error: {str(e)}")
            results.append({
                "id": item["id"],
                "error": str(e)
            })

    # Summary
    avg_latency = total_latency / total_queries if total_queries else 0
    accuracy = (correct_keywords / total_queries) * 100 if total_queries else 0
    
    print("\n" + "="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total Queries Processed : {total_queries}")
    print(f"Average Latency (sec)   : {avg_latency:.2f}")
    print(f"Keyword Accuracy        : {accuracy:.1f}%")
    
    # Save results
    out_path = Path("evaluation/results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeadlockTutorLLM")
    parser.add_argument("--dataset", type=str, default="evaluation/dataset.json", help="Path to evaluation JSON dataset")
    args = parser.parse_args()
    
    evaluate(args.dataset)
