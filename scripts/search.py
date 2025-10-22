# scripts/search.py
import sys, json
from model.decision_tree import NannyDecisionTree
from model.search import search_nannies
from config import MODEL_PATH

def main():
    if len(sys.argv) < 2:
        print("Uso: python -m scripts.search payload.json")
        return
    payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    nannies = payload.get("nannies", [])
    filters = {
        "career": payload.get("career", []),
        "courses": payload.get("courses", []),
        "qualities": payload.get("qualities", []),
        "availability": payload.get("availability"),
        "address": payload.get("address", {})
    }
    
    if isinstance(filters["career"], str):
       filters["career"] = [filters["career"]]
        
    model = NannyDecisionTree()
    model.load(MODEL_PATH)
    results = search_nannies(nannies, filters, model=model)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
