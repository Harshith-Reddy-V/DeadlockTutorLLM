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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.orchestrator import DeadlockTutorOrchestrator, ChatRequest
from backend.llm.provider import get_llm_provider
from backend.rag.retriever import BaseRetriever
from backend.solver.models import (
    BankerStateInput,
    ResourceRequestInput,
    SingleInstanceDetectionInput,
    MultiInstanceDetectionInput,
)


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
            # Build ChatRequest with optional structured solver inputs if present in the benchmark item
            req_kwargs = {"query": query}
            if "banker_state" in item:
                req_kwargs["banker_state"] = BankerStateInput(**item["banker_state"])
            if "resource_request" in item:
                req_kwargs["resource_request"] = ResourceRequestInput(**item["resource_request"])
            if "single_instance_graph" in item:
                req_kwargs["single_instance_graph"] = SingleInstanceDetectionInput(**item["single_instance_graph"])
            if "multi_instance_detection" in item:
                req_kwargs["multi_instance_detection"] = MultiInstanceDetectionInput(**item["multi_instance_detection"])

            req = ChatRequest(**req_kwargs)
            response = orchestrator.handle(req)
            latency = time.time() - start_time
            total_latency += latency
            
            # Combine LLM answer, deterministic worked steps, teaching notes,
            # solver results, and graph diagrams into the comprehensive evaluation text
            eval_corpus = [response.answer]
            if response.worked_steps:
                eval_corpus.extend(response.worked_steps)
            if response.teaching_notes:
                eval_corpus.extend(response.teaching_notes)
            if response.solver_result:
                eval_corpus.append(json.dumps(response.solver_result))
            if response.graph_diagram:
                eval_corpus.append(response.graph_diagram)

            full_eval_text = " ".join(eval_corpus).lower()
            
            # Check keywords across comprehensive response text
            keywords_found = [kw for kw in expected_keywords if kw.lower() in full_eval_text]
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
