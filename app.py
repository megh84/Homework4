import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

st.title("3D Point Cloud Feasibility Analysis App")

uploaded_file = st.file_uploader("Upload dataset_summary.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    y = df["class"].map({"feasible": 1, "infeasible": 0})

    X = df[[
        "points","x_range","y_range","z_range",
        "volume","std_x","std_y","std_z"
    ]].copy()

    eps = 1e-9
    X["point_density"] = X["points"] / (X["volume"] + eps)
    X["aspect_xy"] = X["x_range"] / (X["y_range"] + eps)
    X["aspect_xz"] = X["x_range"] / (X["z_range"] + eps)
    X["aspect_yz"] = X["y_range"] / (X["z_range"] + eps)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipelines = {
        "LogReg": Pipeline([
            ("imp", SimpleImputer()),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "SVM": Pipeline([
            ("imp", SimpleImputer()),
            ("sc", StandardScaler()),
            ("clf", SVC(kernel="rbf"))
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer()),
            ("clf", RandomForestClassifier(n_estimators=200))
        ]),
        "ExtraTrees": Pipeline([
            ("imp", SimpleImputer()),
            ("clf", ExtraTreesClassifier(n_estimators=200))
        ]),
        "AdaBoost": Pipeline([
            ("imp", SimpleImputer()),
            ("clf", AdaBoostClassifier(n_estimators=200))
        ])
    }

    if st.button("Run Pipelines"):
        results = []
        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            f1 = f1_score(y_test, pred)
            results.append({"Model": name, "F1 Score": round(f1,4)})

        res_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)

        st.subheader("Model Comparison (F1 Score)")
        st.dataframe(res_df)

        best_model = res_df.iloc[0]["Model"]
        st.success(f"Best Model: {best_model}")
