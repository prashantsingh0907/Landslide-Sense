"""
Evaluation Metrics and Diagnostic Visualizations for LandslideSense
Calculates confusion matrix, precision, recall, F1-score, ROC-AUC (only when mathematically valid),
and generates Plotly diagnostic figures.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
import plotly.graph_objects as go
import plotly.express as px


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics on the held-out test split.
    Guards mathematically against single-class evaluation splits and zero division.
    """
    y_true = np.array(y_test)
    y_pred = model.predict(X_test)

    # Probabilities for positive class (landslide = 1)
    has_proba = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test)[:, 1] if has_proba else None

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Precision, Recall, F1 with zero_division handling
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    # ROC-AUC is mathematically defined only when both positive and negative samples exist in y_true
    unique_test_classes = np.unique(y_true)
    roc_auc: Optional[float] = None
    fpr_list: List[float] = []
    tpr_list: List[float] = []
    thresholds_list: List[float] = []

    if len(unique_test_classes) > 1 and y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            fpr_list = [float(x) for x in fpr]
            tpr_list = [float(x) for x in tpr]
            thresholds_list = [float(x) for x in thresholds]
        except Exception:
            roc_auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "roc_curve": {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": thresholds_list,
        },
        "test_sample_count": len(y_true),
        "positive_test_count": int(np.sum(y_true == 1)),
        "negative_test_count": int(np.sum(y_true == 0)),
    }


def create_confusion_matrix_chart(cm_dict: Dict[str, Any]) -> go.Figure:
    """Creates an annotated Plotly heatmap for the confusion matrix."""
    matrix = cm_dict.get("matrix", [[0, 0], [0, 0]])
    labels = ["Stable (0)", "Landslide (1)"]
    
    text_annotations = [
        [f"TN: {matrix[0][0]}", f"FP: {matrix[0][1]}"],
        [f"FN: {matrix[1][0]}", f"TP: {matrix[1][1]}"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text_annotations,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorscale=[[0, "#1e293b"], [0.5, "#0284c7"], [1, "#0369a1"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title="Confusion Matrix (Evaluation Split)",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
    )
    return fig


def create_roc_curve_chart(roc_dict: Dict[str, Any], roc_auc: Optional[float]) -> go.Figure:
    """Creates a Plotly ROC curve chart with diagonal reference."""
    fig = go.Figure()
    fpr = roc_dict.get("fpr", [])
    tpr = roc_dict.get("tpr", [])

    if fpr and tpr:
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines+markers",
                name=f"Random Forest (AUC = {roc_auc:.3f})" if roc_auc else "Model ROC",
                line=dict(color="#0284c7", width=3),
            )
        )
    
    # 45-degree diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline (AUC = 0.5)",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
    )
    return fig


def create_feature_importance_chart(
    feature_names: List[str],
    importances: List[float]
) -> go.Figure:
    """Plots relative feature importances derived from the fitted Random Forest."""
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Gini Feature Importances",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        coloraxis_showscale=False,
    )
    return fig
