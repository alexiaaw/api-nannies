import pandas as pd
import argparse
import os
from model.decision_tree import NannyDecisionTree
from config import MODEL_PATH, generate_label

def normalize_df(df):
    
    for col in ["zone", "career", "courses", "qualities"]:
        if col not in df.columns:
            df[col] = None
    
    def parse_list(x):
        if pd.isna(x): return []
        if isinstance(x, list): return x
        if isinstance(x, str):
            try:
                import json
                v = json.loads(x)
                if isinstance(v, list): return v
            except Exception:
                pass
            return [s.strip() for s in x.split(",") if s.strip()]
        return []
        
    df["courses"] = df["courses"].apply(parse_list)
    df["qualities"] = df["qualities"].apply(parse_list)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="ruta a CSV")
    p.add_argument("--json", help="ruta a JSON")
    args = p.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.json:
        df = pd.read_json(args.json)
    else:
        raise RuntimeError("Proveer --csv o --json con datos exportados desde Laravel")

    df = normalize_df(df)
    
    
    feature_cols = ["zone", "career", "courses", "qualities"]
    X = df[feature_cols]
    
    
    y = df.apply(generate_label, axis=1)

    model = NannyDecisionTree()
    model.fit(X, y)
    model.save(MODEL_PATH)
    print("Modelo entrenado y guardado en:", MODEL_PATH)

if __name__ == "__main__":
    main()