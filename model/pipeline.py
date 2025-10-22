from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["zone", "career"]
NUMERIC_COLS = [] 
MULTI_LABEL_COLS = ["courses", "qualities"]

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlbs = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in MULTI_LABEL_COLS:
            mlb = MultiLabelBinarizer()
            
            lists = df.get(col, pd.Series([[]]*len(df))).apply(lambda v: v if isinstance(v, (list, tuple)) else [])
            mlb.fit(lists)
            self.mlbs[col] = mlb
        return self

    def transform(self, X):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        parts = []
        for col in MULTI_LABEL_COLS:
            mlb = self.mlbs.get(col)
            lists = df.get(col, pd.Series([[]]*len(df))).apply(lambda v: v if isinstance(v, (list, tuple)) else [])
            arr = mlb.transform(lists)
            parts.append(arr)
        if parts:
            return np.hstack(parts) # Devuelve un array NumPy
        return np.empty((len(df), 0))

def build_pipeline():
    # Creamos el ColumnTransformer para las columnas CAT y NUM.
    # Dado que las columnas multi-etiqueta ya se procesarán antes, 
    # este ColumnTransformer solo actúa sobre las columnas restantes.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS), 
        ],
        remainder="drop",
        
    )

    # MultiLabelBinarizerTransformer lea las columnas 'courses'/'qualities' ANTES de que el ColumnTransformer las elimine.
    pipeline = Pipeline(
        steps=[
            # 1. Transformar las listas (Multi-etiqueta) a arrays binarios.
            ("multi", MultiLabelBinarizerTransformer()),
            
            # 2. Transformar el resto de las columnas (CAT/NUM), eliminando las demás.
            ("pre", preprocessor),
            
            # 3. Clasificación final.
            ("tree", DecisionTreeClassifier(random_state=42))
        ]
    )
    return pipeline