import io
import wave

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
# LAYER 1: VOICE BIOMETRIC ENGINE (frame-by-frame analysis)
# =====================================================================
def read_wav_bytes(audio_bytes: bytes):
    """Read a PCM WAV (8/16/24/32-bit, mono or stereo) -> (float32 mono array, sample rate)."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 2 ** 23, v - 2 ** 24, v).astype(np.float32)
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        raise ValueError("Unsupported WAV sample width")

    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, sr


def frame_pitch(frame: np.ndarray, sr: int):
    """Estimate pitch period of one frame with normalized autocorrelation."""
    frame = frame - frame.mean()
    n = len(frame)
    spec = np.fft.rfft(frame, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 1e-9:
        return None
    ac = ac / ac[0]

    lo = int(sr / 400)           # highest pitch considered: 400 Hz
    hi = min(int(sr / 70), n - 2)  # lowest pitch considered: 70 Hz
    if hi <= lo + 1:
        return None

    lag = lo + int(np.argmax(ac[lo:hi]))
    strength = float(ac[lag])
    if strength < 0.3:           # unvoiced / noisy frame
        return None

    # Parabolic interpolation for a sub-sample accurate lag
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return (lag + shift) / sr, strength


class VoiceBiometricEngine:
    def analyze_uploaded_file(self, audio_bytes: bytes) -> dict:
        """Returns pitch, jitter, shimmer, instability and husky factor,
        or {"error": "..."} if the audio can't be analysed."""
        try:
            x, sr = read_wav_bytes(audio_bytes)
        except Exception:
            return {"error": "Could not read this WAV file. Please upload an uncompressed PCM .wav (e.g. 16-bit)."}

        if len(x) < sr * 0.3 or np.max(np.abs(x)) < 1e-6:
            return {"error": "Audio is too short or silent. Record at least 1 second of a sustained 'aaah'."}

        x = x / np.max(np.abs(x))

        # Trim leading/trailing silence, keep at most 3 seconds
        loud = np.where(np.abs(x) > 0.05)[0]
        if len(loud) > 0:
            x = x[loud[0]: loud[-1] + 1]
        x = x[: sr * 3]

        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        periods, strengths, rms_vals = [], [], []

        for start in range(0, len(x) - frame_len, hop):
            frame = x[start: start + frame_len]
            res = frame_pitch(frame, sr)
            if res is None:
                continue
            periods.append(res[0])
            strengths.append(res[1])
            rms_vals.append(float(np.sqrt(np.mean(frame ** 2))))

        if len(periods) < 5:
            return {"error": "No clear voiced pitch found. Please upload a clean sustained vowel sound."}

        periods = np.array(periods)
        rms_vals = np.array(rms_vals)
        mean_period = float(np.mean(periods))

        pitch = 1.0 / float(np.median(periods))
        jitter = float(np.mean(np.abs(np.diff(periods))) / mean_period)
        shimmer = float(np.mean(np.abs(np.diff(rms_vals))) / (np.mean(rms_vals) + 1e-9))
        noise_ratio = 1.0 - float(np.mean(strengths))  # lower periodicity = noisier voice

        instability = min(jitter * 750 + shimmer * 150, 100.0)
        husky = float(np.clip(noise_ratio * 1.5 + jitter * 5 + shimmer * 0.5, 0.0, 1.0))

        return {
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4),
            "instability_score": round(instability, 2),
            "fundamental_pitch_hz": round(pitch, 1),
            "husky_factor": round(husky, 2),
        }


# =====================================================================
# LAYER 2: BASELINE GENERATOR FOR NEW PATIENTS
# =====================================================================
def create_initial_patient_history(seed_id: int):
    rng = np.random.default_rng(seed_id)
    days = 30
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")

    return pd.DataFrame({
        "Date": dates,
        "Heart_Rate_BPM": rng.normal(74, 3, days).round(1),
        "Oxygen_Sat_Percentage": np.clip(rng.normal(98.0, 0.4, days), 85, 100).round(1),
        "Systolic_BP_mmHg": rng.normal(122, 4, days).round(1),
        "Voice_Instability_Index": np.clip(rng.normal(12.2, 1.5, days), 0, 100).round(2),
        "Fundamental_Pitch_Hz": rng.normal(135.0, 5.0, days).round(1),
    }).set_index("Date")


# =====================================================================
# LAYER 3: APP UI & MULTI-PATIENT STATE
# =====================================================================
st.set_page_config(page_title="RPMA Dashboard Panel", page_icon="🏥", layout="wide")

st.title("🏥 Remote Patient Monitoring Agent (RPMA)")
st.markdown("Category: **Remote Patient Monitoring (RPM)** | Systemic Endocrine & Neuro-Respiratory Tracking Framework.")
st.markdown("---")

if "master_registry" not in st.session_state:
    st.session_state.master_registry = {
        "PT-01": {"name": "Sarah Jenkins", "age": 52, "condition": "Thyroid Fluid Accumulation (Hypothyroidism)",
                  "history": create_initial_patient_history(42)},
        "PT-02": {"name": "David Miller", "age": 29, "condition": "General Respiratory Screening",
                  "history": create_initial_patient_history(101)},
    }

# --- SIDEBAR ---
st.sidebar.header("📝 Register New Patient File")
with st.sidebar.form("registration_form", clear_on_submit=True):
    new_name = st.text_input("Full Patient Name:")
    new_age = st.number_input("Age (Years):", min_value=1, max_value=120, value=35)
    new_condition = st.selectbox("Suspected Target Diagnostics Condition:", [
        "Thyroid Fluid Accumulation (Hypothyroidism)",
        "Neuro-Vocal Degradation",
        "General Respiratory Screening",
    ])
    submit_registration = st.form_submit_button("Register Patient")

    if submit_registration and new_name:
        new_id = f"PT-{len(st.session_state.master_registry) + 1:02d}"
        st.session_state.master_registry[new_id] = {
            "name": new_name,
            "age": int(new_age),
            "condition": new_condition,
            "history": create_initial_patient_history(np.random.randint(1, 500)),
        }
        st.sidebar.success(f"Registered {new_name} as ID: {new_id}")

st.sidebar.markdown("---")
st.sidebar.header("📁 Electronic Health Records Selector")
patient_options = {pid: f"{pid} - {meta['name']}" for pid, meta in st.session_state.master_registry.items()}
selected_pid = st.sidebar.selectbox(
    "Active Patient Profile Selection:",
    options=list(patient_options.keys()),
    format_func=lambda x: patient_options[x],
)

active_patient = st.session_state.master_registry[selected_pid]
clinical_history = active_patient["history"]
latest_vitals = clinical_history.iloc[-1].copy()

# --- SECTION 1: PROFILE HEADER ---
col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
with col_meta1:
    st.markdown(f"👤 **Patient Name:**\n### {active_patient['name']}")
with col_meta2:
    st.markdown(f"🎂 **Chronological Age:**\n### {active_patient['age']} Years")
with col_meta3:
    st.markdown(f"🩺 **Monitored Condition Path:**\n`{active_patient['condition']}`")
with col_meta4:
    st.markdown(f"🆔 **Internal Record Key:**\n`{selected_pid}`")

st.markdown("---")

# --- SECTION 2: INPUTS ---
st.subheader("📥 Log New Vital Diagnostics Entry & Audio Sample")
form_col1, form_col2 = st.columns(2)

with form_col1:
    st.markdown("##### 📝 Input Vital Measurements")
    # Keys include the patient id so each patient gets their own widget state
    input_hr = st.number_input("Heart Rate (BPM):", min_value=40, max_value=200,
                               value=int(latest_vitals.get("Heart_Rate_BPM", 72)), step=1,
                               key=f"hr_in_{selected_pid}")
    input_bp = st.number_input("Systolic Blood Pressure (mmHg):", min_value=70, max_value=220,
                               value=int(latest_vitals.get("Systolic_BP_mmHg", 120)), step=1,
                               key=f"bp_in_{selected_pid}")
    input_spo2 = st.slider("Oxygen Saturation Level - SpO2 (%):", min_value=80.0, max_value=100.0,
                           value=float(latest_vitals.get("Oxygen_Sat_Percentage", 98.0)), step=0.1,
                           key=f"spo2_in_{selected_pid}")

with form_col2:
    st.markdown("##### 🔊 Hormonal Acoustic Analysis (Voice Record)")
    uploaded_file = st.file_uploader(
        "Drop patient sustained vowel sound file (.wav) to evaluate Vocal Cord Edema patterns",
        type=["wav"],
    )
    if uploaded_file is not None:
        st.audio(uploaded_file.getvalue(), format="audio/wav")

if st.button("🚀 Commit Log Entry directly to Python Pipeline", use_container_width=True):
    commit_ok = True
    note = ""
    vocal_instability_score = 14.20
    detected_pitch = 130.0
    husky_factor = 0.20

    if uploaded_file is not None:
        # getvalue() always returns the full file (read() can return empty after st.audio)
        audio_results = VoiceBiometricEngine().analyze_uploaded_file(uploaded_file.getvalue())
        if "error" in audio_results:
            st.error(audio_results["error"])
            commit_ok = False
        else:
            vocal_instability_score = audio_results["instability_score"]
            detected_pitch = audio_results["fundamental_pitch_hz"]
            husky_factor = audio_results["husky_factor"]
            note = "Voice values measured from the uploaded audio."
    elif "Thyroid" in active_patient["condition"] or "Hypothyroidism" in active_patient["condition"]:
        vocal_instability_score = 26.40
        detected_pitch = 88.5
        husky_factor = 0.72
        note = "No audio uploaded: using fixed DEMO voice values for this condition."
    else:
        note = "No audio uploaded: using fixed DEMO baseline voice values."

    if commit_ok:
        new_entry_row = pd.DataFrame([{
            "Heart_Rate_BPM": float(input_hr),
            "Oxygen_Sat_Percentage": float(input_spo2),
            "Systolic_BP_mmHg": float(input_bp),
            "Voice_Instability_Index": float(vocal_instability_score),
            "Fundamental_Pitch_Hz": float(detected_pitch),
        }], index=[pd.Timestamp.now()])

        st.session_state.master_registry[selected_pid]["history"] = pd.concat([clinical_history, new_entry_row])
        st.session_state["last_result"] = {
            "patient": active_patient["name"],
            "pitch": detected_pitch,
            "instability": vocal_instability_score,
            "husky": husky_factor,
            "note": note,
        }
        st.toast(f"Data packet merged into record stack for {active_patient['name']}", icon="💾")
        st.rerun()

st.markdown("---")

# =====================================================================
# LAYER 4: CLINICAL STATUS & RISK EVALUATION
# =====================================================================
st.subheader("📋 Automated Clinical Assessment & Health Status Report")

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.caption(f"Last analysis ({r['patient']}): {r['note']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Fundamental Pitch (Hz)", r["pitch"])
    m2.metric("Voice Instability Index", r["instability"])
    m3.metric("Husky Factor", r["husky"])

# Latest vs previous entry, so you can see the change numerically
if len(clinical_history) >= 2:
    prev = clinical_history.iloc[-2]
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Heart Rate", f"{latest_vitals['Heart_Rate_BPM']:.1f}", f"{latest_vitals['Heart_Rate_BPM'] - prev['Heart_Rate_BPM']:+.1f}")
    d2.metric("Systolic BP", f"{latest_vitals['Systolic_BP_mmHg']:.1f}", f"{latest_vitals['Systolic_BP_mmHg'] - prev['Systolic_BP_mmHg']:+.1f}")
    d3.metric("SpO2 %", f"{latest_vitals['Oxygen_Sat_Percentage']:.1f}", f"{latest_vitals['Oxygen_Sat_Percentage'] - prev['Oxygen_Sat_Percentage']:+.1f}")
    d4.metric("Voice Instability", f"{latest_vitals['Voice_Instability_Index']:.2f}", f"{latest_vitals['Voice_Instability_Index'] - prev['Voice_Instability_Index']:+.2f}")
    d5.metric("Pitch (Hz)", f"{latest_vitals['Fundamental_Pitch_Hz']:.1f}", f"{latest_vitals['Fundamental_Pitch_Hz'] - prev['Fundamental_Pitch_Hz']:+.1f}")

HR_MIN, HR_MAX = 60, 100
SPO2_MIN = 95.0
BP_SYS_MIN, BP_SYS_MAX = 90, 140
VOICE_INSTABILITY_THRESHOLD = 20.0

current_hr = latest_vitals.get("Heart_Rate_BPM", 72.0)
current_spo2 = latest_vitals.get("Oxygen_Sat_Percentage", 98.0)
current_bp = latest_vitals.get("Systolic_BP_mmHg", 120.0)
current_voice = latest_vitals.get("Voice_Instability_Index", 14.20)
if pd.isna(current_voice):
    current_voice = 14.20

voice_status = "Normal" if current_voice <= VOICE_INSTABILITY_THRESHOLD else "Bad (Elevated)"

risk_flags = []
if not (HR_MIN <= current_hr <= HR_MAX):
    risk_flags.append(f"Abnormal Heart Rate: {current_hr} BPM (Normal range: {HR_MIN}-{HR_MAX} BPM)")
if current_spo2 < SPO2_MIN:
    risk_flags.append(f"Low Oxygen Saturation (SpO2): {current_spo2}% (Threshold: >={SPO2_MIN}%)")
if not (BP_SYS_MIN <= current_bp <= BP_SYS_MAX):
    risk_flags.append(f"Abnormal Systolic BP: {current_bp} mmHg (Normal range: {BP_SYS_MIN}-{BP_SYS_MAX} mmHg)")
if current_voice > VOICE_INSTABILITY_THRESHOLD:
    risk_flags.append(f"Elevated Voice Instability Index: {current_voice:.2f} (Threshold: <{VOICE_INSTABILITY_THRESHOLD})")

if len(risk_flags) > 0:
    st.error(f"⚠️ **Clinical Alert: Potential Health Issues Detected for {active_patient['name']}**")
    st.markdown(f"**Diagnostic Evaluation:** Patient exhibits indicators associated with **{active_patient['condition']}**.")
    st.markdown("The system telemetry has flagged the following concerns:")
    for flag in risk_flags:
        st.markdown(f"- 🔴 {flag}")
    st.warning("💡 **Recommended Action:** Schedule a clinical follow-up or review historical trends below.")
else:
    st.success(f"✅ **Health Status: Stable & Within Target Parameters ({active_patient['name']})**")
    st.markdown(f"**Diagnostic Evaluation:** No acute markers for **{active_patient['condition']}** detected; acoustic indices and vital signs remain within normal baseline parameters.")
    st.markdown(f"- **Heart Rate:** {current_hr} BPM (Stable)")
    st.markdown(f"- **Oxygen Saturation (SpO2):** {current_spo2}% (Optimal)")
    st.markdown(f"- **Systolic BP:** {current_bp} mmHg (Normal)")
    st.markdown(f"- **Voice Instability Index:** {current_voice:.2f} ({voice_status})")
    st.info("💡 **Recommended Action:** Continue standard remote monitoring schedule.")

# =====================================================================
# SECTION 5: TRENDS (one chart per metric so each has its own scale)
# =====================================================================
st.subheader("📈 Historical Telemetry & Trends")
chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Vitals & Hemodynamics", "Acoustic & Voice Biomarkers", "Raw data"])

with chart_tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Heart Rate (BPM)")
        st.line_chart(clinical_history["Heart_Rate_BPM"])
    with c2:
        st.caption("Systolic BP (mmHg)")
        st.line_chart(clinical_history["Systolic_BP_mmHg"])
    with c3:
        st.caption("SpO2 (%)")
        st.line_chart(clinical_history["Oxygen_Sat_Percentage"])

with chart_tab2:
    v1, v2 = st.columns(2)
    with v1:
        st.caption("Voice Instability Index")
        st.line_chart(clinical_history["Voice_Instability_Index"])
    with v2:
        st.caption("Fundamental Pitch (Hz)")
        st.line_chart(clinical_history["Fundamental_Pitch_Hz"])

with chart_tab3:
    st.dataframe(clinical_history.tail(15), use_container_width=True)

st.caption("Demo only – not a medical diagnostic tool.")
