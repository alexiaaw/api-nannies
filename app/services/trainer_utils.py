"""
Funciones auxiliares para transformar DataFrame en X, y matricial.
"""

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd

def prepare_features_and_target(df):
    """
    Entrada: df con columnas [id, name, qualities(list), courses(list), career(list), zone, availability(bool)]
    Salida: X (numpy array), y (numpy array), transformers (dict)
    """
    # Fit MultiLabelBinarizer por cada lista
    mlb_q = MultiLabelBinarizer(sparse_output=False)
    mlb_c = MultiLabelBinarizer(sparse_output=False)
    mlb_ca = MultiLabelBinarizer(sparse_output=False)

    Q = mlb_q.fit_transform(df["qualities"])
    C = mlb_c.fit_transform(df["courses"])
    CA = mlb_ca.fit_transform(df["career"])

    # zone one-hot (manual, acorde a los valores presentes)
    zones = sorted(list(set(df["zone"].astype(str).unique())))
    zmap = {z: i for i, z in enumerate(zones)}
    Z = np.zeros((len(df), len(zones)), dtype=int)
    for i, z in enumerate(df["zone"].astype(str)):
        Z[i, zmap[z]] = 1

    # opciones para feature column names (útiles para debugging)
    q_cols = [f"q__{c}" for c in mlb_q.classes_]
    c_cols = [f"c__{c}" for c in mlb_c.classes_]
    ca_cols = [f"ca__{c}" for c in mlb_ca.classes_]
    z_cols = [f"zone__{z}" for z in zones]

    # Concatenar X
    X = np.hstack([Q, C, CA, Z])
    # Target: availability (proxy)
    y = df["availability"].astype(int).to_numpy()

    transformers = {
        "mlb_q": mlb_q,
        "mlb_c": mlb_c,
        "mlb_ca": mlb_ca,
        "zone_cols": z_cols,
        "feature_columns": q_cols + c_cols + ca_cols + z_cols
    }
    return X, y, transformers
