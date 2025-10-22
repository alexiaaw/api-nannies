# utils/preprocessing.py
import json
import pandas as pd

def normalize_list_field(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        return [s.strip() for s in x.split(",") if s.strip()]
    return []

def normalize_nannies_list(nannies):
    out = []
    for n in nannies:
        nn = dict(n)  # shallow copy
        
        nn["courses"] = normalize_list_field(nn.get("courses"))
        nn["qualities"] = normalize_list_field(nn.get("qualities"))
        
        val = str(nn.get("availability", "")).lower()
        nn["availability"] = val in ["true", "disponible", "si", "sí", "1", "t"]
        
        nn["zone"] = nn.get("zone") or nn.get("zona") or "No especificada"
        out.append(nn)
    return out