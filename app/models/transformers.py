"""
Transformers para preparar features:
- MultiLabelBinarizer para qualities, courses, career.
- One-hot manual para zone (small vocabulary).
Se guardan junto con el modelo en un joblib dict.
"""

from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import os

def fit_transformers(df):
    """
    Ajusta y retorna un dict con transformers y columnas producidas.
    """
    mlb_qualities = MultiLabelBinarizer(sparse_output=False)
    mlb_courses = MultiLabelBinarizer(sparse_output=False)
    mlb_career = MultiLabelBinarizer(sparse_output=False)

    qualities_mat = mlb_qualities.fit_transform(df["qualities"])
    courses_mat = mlb_courses.fit_transform(df["courses"])
    career_mat = mlb_career.fit_transform(df["career"])

    # zone vocabulary (usar valores normalizados)
    zones = sorted(list(set(df["zone"].dropna().astype(str).unique())))
    # devolver nombres de columnas
    qualities_cols = [f"q__{c}" for c in mlb_qualities.classes_]
    courses_cols = [f"c__{c}" for c in mlb_courses.classes_]
    career_cols = [f"ca__{c}" for c in mlb_career.classes_]
    zone_cols = [f"zone__{z}" for z in zones]

    transformers = {
        "mlb_qualities": mlb_qualities,
        "mlb_courses": mlb_courses,
        "mlb_career": mlb_career,
        "zone_vocab": zones,
        "feature_columns": qualities_cols + courses_cols + career_cols + zone_cols + ["availability_input"]  # availability_input included if needed
    }
    return transformers
