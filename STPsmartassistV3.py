import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
import pandas as pd

# ---------------------------------------------------------
# 1. KNOWLEDGE BASE (MSIG VOL 4)
# ---------------------------------------------------------
MSIG_KNOWLEDGE = {
    "FOAM_WHITE": {
        "Diagnosis": "Young Sludge / High F:M Ratio",
        "MSIG_Ref": "Vol 4, Section 5.8.3",
        "Action": "Increase Sludge Age (MCRT) by reducing wasting (WAS)."
    },
    "FOAM_BROWN": {
        "Diagnosis": "Old Sludge / Nocardia Growth",
        "MSIG_Ref": "Vol 4, Section 5.12",
        "Action": "Increase wasting rate (WAS) and check for grease influent."
    },
    "DARK_SEPTIC": {
        "Diagnosis": "Anaerobic Conditions / Low DO",
        "MSIG_Ref": "Vol 4, Section 5.8.2",
        "Action": "Increase blower output. Target DO > 2.0 mg/L."
    },
    "SYSTEM_OK": {
        "Diagnosis": "Normal Operation",
        "MSIG_Ref": "Vol 4, Table 3.2",
        "Action": "Maintain routine inspection."
    }
}

# ---------------------------------------------------------
# 2. IMAGE ANALYSIS
# ---------------------------------------------------------
def extract_features(pil_image):
    image = np.array(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return {
        "foam": np.sum(edges > 0) / edges.size,
        "brightness": np.mean(gray),
        "dark_sludge": np.sum(gray < 40) / gray.size,
        "debug_mask": edges
    }

def msig_inference_engine(features, foam_trigger=0.15):
    if features["dark_sludge"] > 0.45:
        return MSIG_KNOWLEDGE["DARK_SEPTIC"]
    if features["foam"] > foam_trigger:
        if features["brightness"] > 180:
            return MSIG_KNOWLEDGE["FOAM_WHITE"]
        else:
            return MSIG_KNOWLEDGE["FOAM_BROWN"]
    return MSIG_KNOWLEDGE["SYSTEM_OK"]

# ---------------------------------------------------------
# 3. PROCESS ENGINE (NEW)
# ---------------------------------------------------------
def process_inference_engine(data):
    findings = []
    actions = []

    sv30 = data["SV30"]
    do = data["DO"]
    mlss = data["MLSS"]
    nh3 = data["NH3"]
    odour = data["ODOUR"]

    svi = sv30 / mlss * 1000 if mlss > 0 else 0

    if do < 1.5:
        findings.append("Low DO (Possible Septic Condition)")
        actions.append("Increase aeration immediately (Target DO > 2.0 mg/L)")

    if svi > 150:
        findings.append("Bulking Sludge (Poor Settling)")
        actions.append("Check filamentous bacteria, increase RAS")

    elif svi < 80:
        findings.append("Young Sludge (Low Biomass)")
        actions.append("Reduce WAS to increase sludge age")

    if nh3 > 10:
        findings.append("Incomplete Nitrification")
        actions.append("Increase aeration + check MCRT")

    if odour == "Septic (Rotten Egg)":
        findings.append("Anaerobic Condition")
        actions.append("Increase DO + eliminate dead zones")

    elif odour == "Pungent/Ammonia":
        findings.append("High Ammonia Load")
        actions.append("Check influent + nitrification")

    if not findings:
        findings.append("Process Stable")
        actions.append("Maintain operation")

    return {
        "findings": findings,
        "actions": actions,
        "SVI": round(svi, 2)
    }

# ---------------------------------------------------------
# 4. HYDRAULIC CALC
# ---------------------------------------------------------
def calculate_tdh(static_head, flow_lps, pipe_dia_mm, pipe_len_m):
    C = 140
    Q = flow_lps / 1000
    D = pipe_dia_mm / 1000
    hf = 10.67 * (Q/C)**1.852 * (D**-4.87) * pipe_len_m
    return round(static_head + (hf * 1.1), 2)

# ---------------------------------------------------------
# 5. WIZARD (UPDATED)
# ---------------------------------------------------------
def stp_wizard():
    st.sidebar.markdown("---")
    st.sidebar.header("🧙 Troubleshooting Wizard")

    settle = st.sidebar.selectbox(
        "Sludge Settlement:",
        ["Select...", "Settles fast, leaves cloudy water", "Settles slowly, stays suspended", "Plumes/Clumps rising to top"]
    )

    texture = st.sidebar.selectbox(
        "Texture:",
        ["Select...", "Leathery/Thick Brown", "Crisp/White/Bubbly", "Greasy/Oily"]
    )

    st.session_state['wizard_settle'] = settle
    st.session_state['wizard_texture'] = texture

# ---------------------------------------------------------
# 6. FINAL CONSENSUS ENGINE (UPGRADED)
# ---------------------------------------------------------
def final_action_plan(visual_diag, process_result, settle, texture):

    st.write("---")
    st.header("⚡ Integrated Action Plan")

    findings = process_result["findings"]

    if "Anaerobic" in str(findings) or "Low DO" in str(findings):
        st.error("🚨 CRITICAL: Septic Condition")
        st.write("Increase aeration immediately + inspect system")

    elif "Bulking Sludge" in str(findings):
        st.warning("⚠️ Bulking Sludge")
        st.write("Increase RAS + control filamentous growth")

    elif "Young Sludge" in visual_diag or "Young Sludge" in str(findings):
        st.info("ℹ️ Young Sludge")
        st.write("Reduce WAS to build biomass")

    else:
        st.success("✅ System Stable")

# ---------------------------------------------------------
# 7. UI
# ---------------------------------------------------------
st.set_page_config(page_title="MSIG Smart Assist", layout="wide")
st.title("🌊 MSIG Smart Assist Pro")

# Sidebar Inputs
st.sidebar.header("🛠️ Design Settings")
s_head = st.sidebar.number_input("Static Lift (m)", value=5.0)
p_len = st.sidebar.number_input("Pipe Length (m)", value=50.0)
p_dia = st.sidebar.number_input("Pipe Dia (mm)", value=100)
flow = st.sidebar.number_input("Flow (L/s)", value=10.0)

# Process Inputs
st.sidebar.header("🧪 Process Parameters")
sv30 = st.sidebar.number_input("SV30", value=250)
do = st.sidebar.number_input("DO", value=2.0)
mlss = st.sidebar.number_input("MLSS", value=3000)
nh3 = st.sidebar.number_input("NH3", value=5.0)
odour = st.sidebar.selectbox("Odour", ["None", "Earthy", "Septic (Rotten Egg)", "Pungent/Ammonia"])

st.session_state['process_data'] = {
    "SV30": sv30,
    "DO": do,
    "MLSS": mlss,
    "NH3": nh3,
    "ODOUR": odour
}

# Run Wizard
stp_wizard()

# Tabs
tab1, tab2 = st.tabs(["📸 Visual", "📊 Report"])

# ---------------------------------------------------------
# TAB 1
# ---------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        features = extract_features(img)
        diagnosis = msig_inference_engine(features)

        st.session_state['visual_diag'] = diagnosis['Diagnosis']

        col1, col2 = st.columns(2)
        col1.image(img, use_container_width=True)
        col2.image(features["debug_mask"], use_container_width=True)

        st.subheader(diagnosis['Diagnosis'])
        st.info(diagnosis['Action'])

# ---------------------------------------------------------
# TAB 2
# ---------------------------------------------------------
with tab2:
    tdh = calculate_tdh(s_head, flow, p_dia, p_len)

    st.metric("TDH", f"{tdh} m")

    process_result = process_inference_engine(st.session_state['process_data'])

    st.subheader("🧠 Process Analysis")
    st.write(process_result["findings"])
    st.write(process_result["actions"])
    st.metric("SVI", process_result["SVI"])

    # FINAL ENGINE CALL
    if 'visual_diag' in st.session_state:
        final_action_plan(
            st.session_state['visual_diag'],
            process_result,
            st.session_state.get('wizard_settle', ""),
            st.session_state.get('wizard_texture', "")
        )