import streamlit as st
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Note: This assumes your PredictionHead and ConceptDecoder classes are importable
from models import PredictionHead

def load_model(path, params):
    model = PredictionHead(params)
    # model.load_state_dict(torch.load(path)) # Uncomment when you have a saved model
    model.eval()
    return model

def run_dashboard(model, node_types):
    st.set_page_config(layout="wide", page_title="Meta-Path Concept Explorer")
    st.title("🔍 Meta-Path Concept Explorer")
    
    vocab = node_types + ["∅ (STOP)"]
    
    # --- Sidebar: Concept Selection ---
    with torch.no_grad():
        rel_p, time_p, gt_p, gf_p, feat_p, tau_p = model.concept_decoder(model.concepts)
        weights = F.softplus(model.concept_weights)

    st.sidebar.header("Global Concept Gallery")
    concept_idx = st.sidebar.selectbox(
        "Select Concept ID", 
        options=range(model.concepts.shape[0]),
        format_func=lambda x: f"Concept {x} (Weight: {weights[x]:.2f})"
    )

    # --- Header Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction Weight", f"{weights[concept_idx]:.4f}")
    col2.metric("Saturation (Tau)", f"{tau_p[concept_idx].item():.3f}")
    col3.metric("Max Hops", f"{rel_p.shape[1] - 1}")

    st.divider()

    # --- Relational Distribution View ---
    st.header("1. Full Relational Distribution")
    st.write("Visualizing the probability mass assigned to each node type at every step.")
    
    # Prepare data for heatmap/bar charts
    # rel_p shape: [num_concepts, L, R+1]
    dist_data = rel_p[concept_idx].cpu().numpy()
    
    fig_rel = go.Figure()
    for h in range(dist_data.shape[0]):
        step_label = "Root" if h == 0 else f"Hop {h}"
        fig_rel.add_trace(go.Bar(
            name=step_label,
            x=vocab,
            y=dist_data[h],
            text=[f"{v:.2f}" if v > 0.05 else "" for v in dist_data[h]],
            textposition='auto',
        ))
    
    fig_rel.update_layout(
        barmode='group', 
        xaxis_title="Node Type", 
        yaxis_title="Probability Mass",
        yaxis=dict(range=[0, 1.1])
    )
    st.plotly_chart(fig_rel, use_container_width=True)

    # --- Temporal and Feature Windows ---
    st.header("2. Interpretability Windows (Gamma)")
    
    col_t, col_f = st.columns(2)
    
    with col_t:
        st.subheader("Temporal Alignment")
        t_vals = time_p[concept_idx].cpu().numpy()
        t_gams = gt_p[concept_idx].cpu().numpy()
        
        # Plotting the window as an error bar
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=[f"Step {h}" for h in range(len(t_vals))],
            y=t_vals,
            error_y=dict(type='data', array=t_gams, visible=True),
            mode='markers+lines',
            marker=dict(size=12, color='royalblue'),
            name="Target Time"
        ))
        fig_t.update_layout(yaxis_title="Time relative to Root", title="Temporal Window (±γ)")
        st.plotly_chart(fig_t)

    with col_f:
        st.subheader("Feature Variance")
        # Visualizing the scale of gamma for features
        f_gams = gf_p[concept_idx].cpu().numpy()
        fig_f = go.Figure(go.Bar(
            x=[f"Step {h}" for h in range(len(f_gams))],
            y=f_gams,
            marker_color='indianred'
        ))
        fig_f.update_layout(yaxis_title="Gamma (Tolerance)", title="Feature Tightness (Lower = Stricter)")
        st.plotly_chart(fig_f)

    # --- Step-by-Step Raw Table ---
    st.header("3. Raw Prototype Definition")
    steps = []
    for h in range(dist_data.shape[0]):
        top_type = vocab[np.argmax(dist_data[h])]
        steps.append({
            "Step": "ROOT" if h == 0 else f"HOP {h}",
            "Primary Type": top_type,
            "Relational Mass": f"{np.max(dist_data[h]):.2f}",
            "Time": f"{t_vals[h]:.1f} (±{t_gams[h]:.1f})",
            "Feature Gamma": f"{f_gams[h]:.3f}"
        })
    st.table(pd.DataFrame(steps))