import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from config import DEVICE, OUTPUT_DIR, UNIFIED_DIR, dashboard_config
from models.simulation.trajectory import TrajectoryGenerator
from pipeline.digital_twin import DigitalTwin
from pipeline.inference import InferencePipeline, PipelineResult

st.set_page_config(
    page_title=dashboard_config.page_title,
    page_icon=dashboard_config.page_icon,
    layout=dashboard_config.layout,
    initial_sidebar_state=dashboard_config.initial_sidebar_state,
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low { background-color: #4CAF50; }
    .risk-moderate { background-color: #FF9800; }
    .risk-high { background-color: #f44336; }
    .risk-critical { background-color: #9C27B0; }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "result" not in st.session_state:
    st.session_state.result = None
if "digital_twin" not in st.session_state:
    st.session_state.digital_twin = None
if "trajectory_generator" not in st.session_state:
    st.session_state.trajectory_generator = None


@st.cache_resource
def load_pipeline():
    pipeline = InferencePipeline()
    pipeline.load()
    return pipeline


@st.cache_resource
def load_trajectory_generator():
    gen = TrajectoryGenerator()
    gen.load()
    return gen


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)


def get_risk_color(risk_level: str) -> str:
    colors = {
        "low": "#4CAF50",
        "moderate": "#FF9800",
        "high": "#f44336",
        "critical": "#9C27B0",
    }
    return colors.get(risk_level, "#666")


with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/hospital.png", width=60)
    st.title("🏥 Digital Twin")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔬 Analyze Wound", "📊 Dashboard", "🎬 Simulate Healing", "📜 History"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    with st.expander("⚙️ Settings"):
        patient_id = st.text_input("Patient ID", value="default_patient")
        wound_id = st.text_input("Wound ID", value="wound_1")
        skip_detection = st.checkbox("Skip Detection (use full image)", value=False)
        st.markdown("---")

        if st.button("🔄 Reset Pipeline"):
            st.session_state.pipeline = None
            st.session_state.result = None
            st.rerun()

    st.markdown("---")
    st.caption(f"Device: {DEVICE}")
    st.caption("v1.0.0 | Made with ❤️")


if page == "🔬 Analyze Wound":
    st.markdown('<h1 class="main-header">🔬 Wound Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a wound image for AI-powered analysis</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Wound Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the wound for analysis",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📁 Load Sample"):
            sample_dir = UNIFIED_DIR / "train"
            if sample_dir.exists():
                samples = list(sample_dir.rglob("*.jpg"))[:5]
                if samples:
                    st.session_state.sample_images = samples

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(numpy_to_pil(image), use_container_width=True)

        if st.button("🚀 Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Loading AI models..."):
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = load_pipeline()

            with st.spinner("Analyzing wound..."):
                progress = st.progress(0)
                result = st.session_state.pipeline.analyze(image, skip_detection=skip_detection)
                st.session_state.result = result
                progress.progress(100)

            if st.session_state.digital_twin is None:
                st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)
            st.session_state.digital_twin.update(result)

            st.success("✅ Analysis complete!")
            st.rerun()

        if st.session_state.result is not None:
            result = st.session_state.result

            with col2:
                st.subheader("🎯 Detection & ROI")
                if result.roi_image is not None:
                    st.image(numpy_to_pil(result.roi_image), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Key Metrics")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                if result.classification:
                    st.metric("Wound Type", result.classification.wound_type.title())
                    st.progress(result.classification.wound_type_confidence)

            with m2:
                if result.classification:
                    st.metric("Severity", result.classification.severity.title())
                    st.progress(result.classification.severity_confidence)

            with m3:
                if result.risk:
                    st.metric("Risk Score", f"{result.risk.risk_score:.2f}")
                    color = get_risk_color(result.risk.risk_level)
                    st.markdown(
                        f'<div style="background: {color}; padding: 0.5rem; border-radius: 5px; text-align: center; color: white;">'
                        f'{result.risk.risk_level.upper()}</div>',
                        unsafe_allow_html=True,
                    )

            with m4:
                if result.features:
                    st.metric("Wound Area", f"{result.features.wound_area:.0f} px²")
                    if result.features.total_volume:
                        st.metric("Volume", f"{result.features.total_volume:.0f}")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(["🧬 Segmentation", "🌊 Depth", "⚠️ Risk", "⏱️ Timing"])

            with tab1:
                if result.segmentation:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if result.segmentation.overlay is not None:
                            st.image(numpy_to_pil(result.segmentation.overlay), caption="Tissue Segmentation", use_container_width=True)

                    with col_b:
                        st.write("**Tissue Composition:**")
                        for tissue, pct in result.segmentation.class_percentages.items():
                            st.progress(pct / 100, text=f"{tissue.title()}: {pct:.1f}%")

                        fig = px.pie(
                            values=list(result.segmentation.class_percentages.values()),
                            names=list(result.segmentation.class_percentages.keys()),
                            title="Tissue Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if result.depth:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(numpy_to_pil(result.depth.depth_colored), caption="Depth Map", use_container_width=True)

                    with col_b:
                        st.metric("Mean Depth", f"{result.depth.mean_depth:.3f}")
                        st.metric("Max Depth", f"{result.depth.max_depth:.3f}")
                        if result.volume:
                            st.write("**Volume by Tissue:**")
                            for tissue, vol in result.volume.tissue_volumes.items():
                                st.write(f"{tissue}: {vol:.2f}")

            with tab3:
                if result.risk:
                    st.subheader(f"Risk Level: {result.risk.risk_level.upper()}")
                    st.metric("Risk Score", f"{result.risk.risk_score:.2f}")

                    st.write("**Recommendations:**")
                    for i, rec in enumerate(result.risk.recommendations, 1):
                        st.write(f"{i}. {rec}")

            with tab4:
                st.write("**Inference Times:**")
                for stage, time_val in result.inference_times.items():
                    st.write(f"- {stage.title()}: {time_val:.3f}s")
                st.metric("Total Time", f"{result.total_time:.2f}s")


elif page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 Patient Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Track wound healing progress over time</p>', unsafe_allow_html=True)

    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)

    twin = st.session_state.digital_twin

    if len(twin.states) == 0:
        st.warning("No wound data available. Please analyze a wound first.")
    else:
        current = twin.get_current_state()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Measurements", len(twin.states))
        with col2:
            st.metric("Wound Type", current.wound_type.title())
        with col3:
            st.metric("Current Risk", f"{current.risk_score:.2f}")
        with col4:
            color = get_risk_color(current.risk_level)
            st.markdown(
                f'<div style="background-color: {color}; padding: 0.5rem; border-radius: 5px; text-align: center; color: white;">'
                f'{current.risk_level.upper()}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        if len(twin.states) >= 2:
            trends = twin.compute_trends()
            prediction = twin.predict_healing_time()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📈 Trend Analysis")

                timestamps = [s.timestamp for s in twin.states]
                risk_scores = [s.risk_score for s in twin.states]
                areas = [s.wound_area for s in twin.states]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=timestamps, y=risk_scores, name="Risk Score", mode="lines+markers"))
                fig.update_layout(title="Risk Score Over Time", height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("🔮 Healing Prediction")

                if prediction["status"] == "predicted":
                    st.metric(
                        "Est. Days to Heal",
                        f"{prediction['estimated_days_to_heal']:.0f}",
                        delta=f"Trend: {trends['overall_trend']}",
                    )
                    st.progress(prediction["confidence"])
                    st.caption(f"Confidence: {prediction['confidence']:.0%}")
                else:
                    st.info("More data needed for prediction")

        st.markdown("---")
        st.subheader("📋 Full Report")

        with st.expander("View Report"):
            st.text(twin.generate_report())


elif page == "🎬 Simulate Healing":
    st.markdown('<h1 class="main-header">🎬 Healing Simulation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-generated healing trajectory visualization</p>', unsafe_allow_html=True)

    if st.session_state.result is None:
        st.warning("⚠️ Please analyze a wound first before simulating healing.")
    else:
        result = st.session_state.result

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("⚙️ Simulation Settings")

            current_severity = st.selectbox(
                "Current Severity",
                options=[4, 3, 2, 1, 0],
                format_func=lambda x: ["Healed", "Mild", "Moderate", "Severe", "Critical"][x],
                index=2,
            )

            treatment = st.selectbox("Treatment Scenario", options=["optimal", "standard", "suboptimal"])
            num_frames = st.slider("Simulation Steps", 3, 10, 5)

            if st.button("🎬 Generate Trajectory", type="primary"):
                with st.spinner("Generating healing simulation..."):
                    if st.session_state.trajectory_generator is None:
                        st.session_state.trajectory_generator = TrajectoryGenerator()

                    gen = st.session_state.trajectory_generator

                    frames = gen.generate_trajectory(
                        start_image=(result.roi_image if result.roi_image is not None else result.original_image),
                        start_severity=current_severity,
                        treatment_scenario=treatment,
                        num_days=30,
                        frames_per_day=0.3,
                    )

                    st.session_state.trajectory_frames = frames

                st.success(f"✅ Generated {len(frames)} frames!")

        with col2:
            st.subheader("📽️ Trajectory Preview")

            if "trajectory_frames" in st.session_state and st.session_state.trajectory_frames:
                frames = st.session_state.trajectory_frames

                frame_idx = st.slider("Timeline", 0, len(frames) - 1, 0)
                frame = frames[frame_idx]

                col_a, col_b = st.columns(2)

                with col_a:
                    st.image(numpy_to_pil(frame.image), use_container_width=True)

                with col_b:
                    st.write(f"**Day {frame.day}**")
                    st.write(f"Severity: {frame.severity_name.title()}")

                    if frame.tissue_change:
                        fig = px.bar(
                            x=list(frame.tissue_change.keys()),
                            y=[v * 100 for v in frame.tissue_change.values()],
                            title="Predicted Tissue %",
                        )
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Click 'Generate Trajectory' to simulate healing")


elif page == "📜 History":
    st.markdown('<h1 class="main-header">📜 Analysis History</h1>', unsafe_allow_html=True)

    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = DigitalTwin(patient_id, wound_id)

    twin = st.session_state.digital_twin

    if len(twin.states) == 0:
        st.info("No history available yet. Analyze wounds to build history.")
    else:
        history_data = []
        for i, state in enumerate(twin.states):
            history_data.append(
                {
                    "#": i + 1,
                    "Timestamp": state.timestamp,
                    "Type": state.wound_type,
                    "Severity": state.severity,
                    "Risk Score": f"{state.risk_score:.2f}",
                    "Risk Level": state.risk_level,
                    "Wound Area": f"{state.wound_area:.0f}",
                }
            )

        st.dataframe(history_data, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Export Timeline JSON"):
                path = twin.export_timeline()
                st.success(f"Exported to {path}")

        with col2:
            if st.button("📋 Copy Report"):
                report = twin.generate_report()
                st.code(report)


st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>
        Digital Twin System for Chronic Wound Analysis | 
        Powered by PyTorch & Streamlit | 
        For Research Purposes Only
    </small>
</div>
""",
    unsafe_allow_html=True,
)
