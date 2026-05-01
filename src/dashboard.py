import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add MTP-2/ to path
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
import plotly.express as px
import traceback
import config
from dataset import PTBXLDataset
from models import ECGXAINet
from interpretability import ECGExplainer, GradCAM, BioInspiredExplainer
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Set page config
st.set_page_config(layout="wide", page_title="ECG-XAI-Net Clinical Dashboard", page_icon="🫀")

# Custom CSS for clinical feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    .stAlert {
        padding: 0.5rem;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .blink-red {
        color: #e74c3c;
        font-weight: bold;
        animation: blink 1s linear infinite;
    }
    .bold-red {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED LOADERS ---

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGXAINet(num_leads=config.NUM_LEADS, num_classes=config.NUM_CLASSES)
    
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        st.error(f"Model not found at {model_path}")
        return None, None
        
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_data():
    dataset = PTBXLDataset(config.DATA_DIR, split='test', sampling_rate=config.SAMPLING_RATE, augment_prob=0.0)
    return dataset

# --- HELPER FUNCTIONS ---

def plot_interactive_ecg(signal, lead_names, height=1800, width=1000):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=12, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=lead_names)
    
    time = np.arange(signal.shape[1]) / config.SAMPLING_RATE
    
    for i in range(12):
        fig.add_trace(go.Scatter(x=time, y=signal[i], mode='lines', name=lead_names[i],
                                line=dict(color='#2980b9', width=1.5)),
                     row=i+1, col=1)
        if i < 11:
            fig.update_xaxes(showticklabels=False, row=i+1, col=1)
            
    fig.update_layout(height=height, width=width, showlegend=False,
                      template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    fig.update_xaxes(title_text="Time (s)", row=12, col=1)
    return fig

def render_patient_view(dataset, model, device, sample_idx, unique_key=""):
    # Load sample and metadata
    signal, labels = dataset[sample_idx]
    true_labels = [dataset.classes[i] for i, x in enumerate(labels) if x == 1]
    ecg_id = dataset.records[sample_idx]
    meta_row = dataset.df.loc[ecg_id]
    age = meta_row.get('age', 'N/A')
    sex_val = meta_row.get('sex', 0)
    sex = "Male" if sex_val == 0 else ("Female" if sex_val == 1 else str(sex_val))
    
    # Header
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: st.markdown(f"**ID:** {ecg_id}")
    with c2: st.markdown(f"**Age:** {age}")
    with c3: st.markdown(f"**Sex:** {sex}")
    st.markdown(f"**True Dx:** {', '.join(true_labels) if true_labels else 'Norm'}")
    
    # Prediction
    input_tensor = torch.tensor(signal).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(input_tensor)
        probs = preds.cpu().numpy()[0]
    
    st.subheader("Results")
    class_names = dataset.classes
    threshold = st.session_state.get('threshold', 0.5)
    
    predicted_classes = []
    for i, (cls, prob) in enumerate(zip(class_names, probs)):
        if prob >= threshold:
            st.markdown(f"**{cls}: <span class='bold-red'>{prob:.1%}</span> <span class='blink-red'>(Positive)</span>**", unsafe_allow_html=True)
            predicted_classes.append(cls)
        else:
            st.markdown(f"{cls}: {prob:.1%}") 

    # Prob Chart with Distinct Colors
    prob_df = pd.DataFrame({'Class': class_names, 'Probability': probs})
    color_map = {
        'NORM': '#2ecc71', # Green
        'MI': '#e74c3c',   # Red
        'STTC': '#f39c12', # Orange
        'CD': '#9b59b6',   # Purple
        'HYP': '#3498db'   # Blue
    }
    fig_p = px.bar(prob_df, x='Probability', y='Class', orientation='h', range_x=[0, 1],
                   color='Class', color_discrete_map=color_map)
    fig_p.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_p, width='stretch', key=f"p_{unique_key}")
    
    # --- Inner View Rendering Functions ---
    def show_signals(k):
        leads = config.LEAD_NAMES if hasattr(config, 'LEAD_NAMES') else [f'Lead {i+1}' for i in range(12)]
        fig = plot_interactive_ecg(signal, leads, height=1000)
        st.plotly_chart(fig, width='stretch', key=f"ecg_f_{k}")

    def show_xai(k):
        m = st.selectbox("XAI Method", ["Integrated Gradients", "Grad-CAM", "SHAP"], key=f"xm_{k}")
        t = st.selectbox("Target Class", class_names, key=f"xt_{k}")
        
        if st.checkbox("Show Explanations", key=f"xc_{k}"):
            with st.spinner("Analyzing..."):
                if m == "SHAP":
                    try:
                        explainer = ECGExplainer(model, torch.zeros(1, 12, 5000).to(device), device)
                        shap_vals = explainer.explain(input_tensor)
                        idx = class_names.index(t)
                        s = shap_vals[idx][0] if isinstance(shap_vals, list) else shap_vals[0]
                        if isinstance(s, torch.Tensor): s = s.cpu().detach().numpy()
                        fig = px.line(y=s[1], title=f"SHAP Attribution - Lead II ({t})")
                        st.plotly_chart(fig, width='stretch', key=f"sh_{k}")
                    except Exception as e: st.error(f"SHAP Error: {e}")
                
                elif m == "Grad-CAM":
                    try:
                        grad_cam = GradCAM(model, model.conv4)
                        input_tensor.requires_grad = True
                        cam = grad_cam.generate_cam(input_tensor, target_class=class_names.index(t))
                        f = interp1d(np.linspace(0, 1, len(cam)), cam, kind='linear')
                        cam_up = f(np.linspace(0, 1, 5000))
                        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=signal[1], mode='lines', name='ECG Lead II', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(y=cam_up * 0.5, mode='lines', name='Attention', fill='tozeroy', line=dict(color='red')))
                        fig.update_layout(title=f"Grad-CAM Attention ({t})", height=400)
                        st.plotly_chart(fig, width='stretch', key=f"cam_{k}")
                    except Exception as e: st.error(f"Grad-CAM Error: {e}")

                elif m == "Integrated Gradients":
                    try:
                        ig = BioInspiredExplainer(model, device)
                        attr = ig.explain(input_tensor, target_class=class_names.index(t))
                        attr_l2 = attr[0, 1, :].cpu().detach().numpy() if isinstance(attr, torch.Tensor) else attr[0, 1, :]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=signal[1], mode='lines', name='Signal', line=dict(color='gray', width=0.5)))
                        fig.add_trace(go.Scattergl(y=signal[1], mode='markers', name='Importance', marker=dict(color=attr_l2, colorscale='Turbo', size=3)))
                        fig.update_layout(title=f"Integrated Gradients ({t})", height=400)
                        st.plotly_chart(fig, width='stretch', key=f"ig_{k}")
                    except Exception as e: st.error(f"IG Error: {e}")

    def show_report(k):
        rep_state_key = f"rep_text_{k}"
        if st.button("Generate AI Report", key=f"rep_{k}"):
            report = f"**Patient ID:** {ecg_id}\n\n**Diagnosis:** {', '.join(predicted_classes) if predicted_classes else 'Normal'}\n"
            report += f"**Confidence:** {max(probs):.1%}\n\n**Findings:**\n"
            found = False
            f_map = {'NORM': "- Normal ECG rhythm.\n", 'MI': "- Evidence of Infarction.\n", 'STTC': "- ST/T Abnormalities.\n", 'CD': "- Conduction Disturbance.\n", 'HYP': "- Hypertrophy signs.\n"}
            for cls in predicted_classes:
                report += f_map.get(cls, f"- {cls} detected.\n")
                found = True
            if not found: report += "- No significant findings meeting threshold.\n"
            st.session_state[rep_state_key] = report
            
        if rep_state_key in st.session_state:
            st.markdown(st.session_state[rep_state_key])

    # UI View Switcher
    view_labels = ["Signals", "Interpretability", "Report"]
    
    # Persist Selection in Comparison Mode
    if "left" in unique_key or "right" in unique_key:
        choice = st.radio("Select View:", view_labels, horizontal=True, key=f"sel_{unique_key}")
        if choice == "Signals": show_signals(unique_key)
        elif choice == "Interpretability": show_xai(unique_key)
        elif choice == "Report": show_report(unique_key)
    else:
        # Standard Tabs for Single Mode
        t1, t2, t3 = st.tabs(view_labels)
        with t1: show_signals(f"{unique_key}_t1")
        with t2: show_xai(f"{unique_key}_t2")
        with t3: show_report(f"{unique_key}_t3")

def render_gallery_view():
    st.header("📊 AI Visual Insights Gallery")
    st.markdown("Explore detailed evaluation metrics and XAI visualizations generated during the model analysis phase.")
    
    plot_dir = os.path.join(config.BASE_DIR, 'plots')
    if not os.path.exists(plot_dir):
        st.error("Plots directory not found.")
        return
        
    explanations = {
        "roc_curves.png": {
            "title": "ROC Curves (Diagnostic Discrimination)",
            "desc": """
**Description:** Receiver Operating Characteristic (ROC) curves illustrate the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

**Why is it so?** 
In clinical ECG analysis, we must balance 'Sensitivity' (catching every MI) and 'Specificity' (avoiding false alarms). 
- The **AUC (Area Under Curve)** score represents the probability that the model will rank a randomly chosen positive instance higher than a randomly chosen negative one.
- **Clinical Insight:** A curve reaching the top-left corner indicates a 'Gold Standard' performance, meaning the model can distinguish between conditions (like MI vs. NORM) with near-perfect accuracy without being fooled by noise.
            """
        },
        "confusion_matrix.png": {
            "title": "Confusion Matrix (Reliability Map)",
            "desc": """
**Description:** A visualization of the performance of an algorithm. Each row represents the instances in an actual class while each column represents the instances in a predicted class.

**Why is it so?** 
It highlights the 'Blind Spots' of the AI. For example, if the model often confuses **CD (Conduction Disturbance)** with **NORM**, it tells us that the conduction delays are too subtle for the current feature set.
- **Clinical Insight:** Diagonal cells represent correct diagnoses. Off-diagonal cells tell surgeons where human oversight is most critical. If MI is being misclassified as NORM, the 'Cost of Error' is high, requiring manual review.
            """
        },
        "shap_example.png": {
            "title": "SHAP Global Importance (The 'Reasoning' Map)",
            "desc": """
**Description:** SHAP (Shapley Additive Explanations) assigns each feature an importance value for a particular prediction. This plot shows the global average importance.

**Why is it so?** 
Deep learning is often a 'black box'. SHAP breaks it open by showing exactly which leads (e.g., V1-V6 vs. Leads I-III) the model relies on for its decisions.
- **Clinical Insight:** If SHAP shows the model is ignoring Lead II but heavily weighting V1 for an MI diagnosis, it confirms the AI is looking at the 'Anterior' part of the heart—aligning with how a cardiologist would diagnose an Anterior Wall MI.
            """
        },
        "grad_cam_example.png": {
            "title": "Grad-CAM Attribution (Morphological Focus)",
            "desc": """
**Description:** Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of the target class flowing into the final convolutional layer to produce a localization map.

**Why is it so?** 
While SHAP tells us 'which lead', Grad-CAM tells us 'which part of the wave'. It highlights the **P-wave, QRS complex, or T-wave**.
- **Clinical Insight:** For ST-segment elevation (MI), the Grad-CAM heat should ideally center on the ST segment. If it highlights the P-wave instead, the model might be learning 'shortcut' noise rather than clinical pathology.
            """
        },
        "deeplift_example.png": {
            "title": "Integrated Gradients / DeepLIFT (High Res Evidence)",
            "desc": """
**Description:** This method assigns an 'attribution' score to every single data point in the signal by comparing it to a baseline (zero signal).

**Why is it so?** 
It provides the highest resolution of 'evidence'. Every spike in the coloring represents a specific point in time where the model's confidence jumped.
- **Clinical Insight:** It acts like a digital 'highlighter'. Surgeons can see if the 'pathological notch' they see in the signal is the same notch the AI used to make its decision.
            """
        },
        "attention_map_example.png": {
            "title": "Transformer Attention Map (Global Context)",
            "desc": """
**Description:** Visualizes the self-attention weights from the Transformer layers. It shows how the model 'connects' different parts of the ECG across time.

**Why is it so?** 
Unlike standard CNNs that look at small windows, the Transformer 'Attends' to the whole 10-second strip. This mimics how a doctor looks at the 'regularity' of beats across the entire page.
- **Clinical Insight:** High attention on the relationship between distant QRS complexes indicates the model is checking for Rhythmicity (detecting Arrhythmias or Conduction blocks).
            """
        },
        "calibration_curves.png": {
            "title": "Probability Calibration (Trustworthiness)",
            "desc": """
**Description:** A calibration curve shows how well the probabilistic predictions of a classifier are calibrated.

**Why is it so?** 
If a model says '90% confidence', it should be right 9 out of 10 times. If it's 90% confident but wrong half the time, it's 'Overconfident' and dangerous for clinical use.
- **Clinical Insight:** This plot tells doctors *when to trust the percentage*. If the curve follows the 45-degree line, the AI's confidence levels are 'Clinically Honest'.
            """
        },
        "error_analysis.png": {
            "title": "Automated Error Analysis (Structural Weakness)",
            "desc": """
**Description:** A statistical breakdown of error rates across different patient cohorts (Age, Sex, Heart Rate).

**Why is it so?** 
AI models can be biased. This plot reveals if the model performs worse on elderly patients or specific genders.
- **Clinical Insight:** Identifying 'High Error Zones' (e.g., poor performance in patients > 80 years old) allows clinicians to apply extra caution when use of AI is involved in those specific cases.
            """
        },
        "precision_recall_curves.png": {
            "title": "Precision-Recall (Detection Quality)",
            "desc": """
**Description:** Displays the trade-off between Precision (exactness) and Recall (completeness) for different thresholds.

**Why is it so?** 
In ECG screening, high **Recall** is vital (don't miss a heart attack!), but low **Precision** leads to 'Alarm Fatigue' (too many false positives).
- **Clinical Insight:** These curves help hospital administrators set the 'Optimal Alert Threshold' for the dashboard to minimize missed cases while keeping the ICU quiet.
            """
        },
        "benchmark_comparison_graph.png": {
            "title": "Model Benchmarking (Architecture Validation)",
            "desc": """
**Description:** A bar chart comparing the performance of our Hybrid ECG-XAI-Net against standard models like ResNet or InceptionTime.

**Why is it so?** 
It proves that the 'CNN + Transformer' approach is actually better than standard 'Image-based' AI for 1-D signals.
- **Clinical Insight:** Demonstrates the value of using specialized clinical architecture over generic 'off-the-shelf' AI, justifying the research complexity.
            """
        },
        "lime_explanation.png": {
            "title": "LIME Local Explanation (Simplification)",
            "desc": """
**Description:** LIME creates a locally faithful linear model around a single prediction to explain why that specific case was classified as it was.

**Why is it so?** 
It's like a 'simplified second opinion'. It tells you: 'For THIS specific patient, if the voltage in Lead V1 was slightly lower, the diagnosis would have changed.'
- **Clinical Insight:** Excellent for explaining individual 'Edge Cases' where the model's decision seems surprising to a human observer.
            """
        },
        "performance_comparison_table.png": {
            "title": "Performance vs. State-of-the-Art Targets",
            "desc": """
**Description:** A comprehensive comparison table showing how ECG-XAI-Net performs against established benchmarks from recent literature.

**Why is it so?** 
This validates the model's clinical readiness by comparing across 5 key metrics: Superclass AUC (overall accuracy), Binary Accuracy (class-specific), AFib AUROC (critical arrhythmia detection), Rare Class Sensitivity (ability to catch unusual conditions), and XAI Methods (explainability).

- **Clinical Insight:** The model achieves 90.6% AUC (97.4% of the 93% target) and EXCEEDS XAI integration goals (5 methods vs. 4 target). The main gap is in rare class sensitivity (40.4% vs. 85% target for Hypertrophy), which is a known challenge in imbalanced medical datasets. This transparency helps clinicians understand where the AI excels and where human oversight is critical.
            """
        }
    }
    
    visual_order = [
         "performance_comparison_table.png",
         "benchmark_comparison_graph.png",
         "grad_cam_example.png",
         "attention_map_example.png",
         "shap_example.png",
         "deeplift_example.png",
         "lime_explanation.png",
         "roc_curves.png",
         "precision_recall_curves.png",
         "error_analysis.png",
         "calibration_curves.png",
         "confusion_matrix.png"
    ]
    
    # Filter files and sort based on predefined order
    available_files = [f for f in os.listdir(plot_dir) if f.endswith('.png') and f in explanations]
    
    # Sort: First those in visual_order, then others alphabetically
    files = sorted(available_files, key=lambda x: visual_order.index(x) if x in visual_order else len(visual_order) + 1)
    
    if not files:
        st.info("No insight plots available in the project directory.")
        return
        
    # Select Visual to View (Default: First item, which is performance table per sort)
    visual_names = files 
    default_index = 0
    
    selected_visual_file = st.selectbox(
        "Select a Visualization to Analyze:",
        options=visual_names,
        format_func=lambda x: explanations[x]["title"],
        index=default_index
    )
    
    if selected_visual_file:
        img_path = os.path.join(plot_dir, selected_visual_file)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(img_path, width='stretch')
        with c2:
            st.subheader("Clinical Detail")
            st.info(explanations[selected_visual_file]['desc'])
            st.markdown("---")
            st.markdown("**Analysis Metadata:**")
            st.write(f"- **Filename:** `{selected_visual_file}`")
            st.write(f"- **Source:** Multi-lead ECG-XAI-Net evaluation pipeline")

# --- MAIN APP ---

def main():
    st.title("🫀 ECG-XAI-Net: Clinical Dashboard")
    
    model, device = load_model()
    dataset = load_data()
    if model is None or dataset is None: st.stop()
        
    st.sidebar.header("Dashboard Mode")
    mode = st.sidebar.radio("View Mode", ["Single Patient", "Compare (Male vs Female)", "Visual Insights Gallery"])
    threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.session_state['threshold'] = threshold
    
    if mode == "Single Patient":
        st.sidebar.header("Patient Selection")
        sel_mode = st.sidebar.radio("Select By:", ["Index", "Random"], key="s_m")
        if sel_mode == "Index":
            s_idx = st.sidebar.number_input("Index", 0, len(dataset)-1, 0)
        else:
            if st.sidebar.button("Pick Random Sample", key="b_s_r"):
                st.session_state['s_idx'] = np.random.randint(0, len(dataset))
                if "rep_text_single" in st.session_state: del st.session_state["rep_text_single"]
            s_idx = st.session_state.get('s_idx', 0)
        render_patient_view(dataset, model, device, s_idx, "single")
        
    elif mode == "Compare (Male vs Female)":
        st.sidebar.header("Comparison Selection")
        if st.sidebar.button("🎲 Random Pair (Male vs Female)", key="b_c_r"):
            idx_m = np.random.randint(0, len(dataset))
            while dataset.df.loc[dataset.records[idx_m]].get('sex', 0) != 0:
                idx_m = np.random.randint(0, len(dataset))
            idx_f = np.random.randint(0, len(dataset))
            while dataset.df.loc[dataset.records[idx_f]].get('sex', 0) != 1:
                idx_f = np.random.randint(0, len(dataset))
            st.session_state['cm'], st.session_state['cf'] = idx_m, idx_f
            if "rep_text_left_m" in st.session_state: del st.session_state["rep_text_left_m"]
            if "rep_text_right_f" in st.session_state: del st.session_state["rep_text_right_f"]
            
        idx_m_comp = st.session_state.get('cm', 0)
        idx_f_comp = st.session_state.get('cf', 1) 
        
        cl, cr = st.columns(2)
        with cl:
            st.header("👨 Male Patient")
            render_patient_view(dataset, model, device, idx_m_comp, "left_m")
        with cr:
            st.header("👩 Female Patient")
            render_patient_view(dataset, model, device, idx_f_comp, "right_f")
    
    else:
        render_gallery_view()

if __name__ == "__main__":
    main()
