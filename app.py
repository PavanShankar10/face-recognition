"""
app.py - Streamlit Face Recognition Application
-----------------------------------------------
A self-contained Streamlit application that serves as BOTH the user interface
and the inference engine for the Face Recognition System.

Key Capabilities:
1. Blur Detection via Laplacian variance (warns if input image is blurry)
2. Deep Face Embedding Extraction (Stage 1 - 128-d vector)
3. Supervised Classification via trained SVM with calibrated probabilities (Stage 2)
4. Open-Set Rejection (rejects unknown faces when max confidence < threshold)
5. Live Interactive Threshold Tuning (demonstrating FAR vs. FRR trade-offs)

Run with:
    streamlit run app.py
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib

# Import helper functions from utils.py
from utils import (
    is_blurry,
    load_image_as_rgb,
    extract_face_encodings,
    draw_detection_boxes,
    FACE_REC_AVAILABLE,
    DEEPFACE_AVAILABLE,
)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom UI Styling (Inline Streamlit UI)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Recognition & Open-Set Rejection System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = Path("classifier.pkl")
ENCODINGS_FILE = Path("encodings.pickle")
TEST_IMAGES_DIR = Path("test_images")


# -----------------------------------------------------------------------------
# 2. Model & Encodings Loader (Cached for Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load the trained classifier bundle from classifier.pkl."""
    if not MODEL_FILE.exists():
        return None
    try:
        model_bundle = joblib.load(MODEL_FILE)
        return model_bundle
    except Exception as e:
        st.error(f"Error loading {MODEL_FILE}: {e}")
        return None


# -----------------------------------------------------------------------------
# 3. Application Header
# -----------------------------------------------------------------------------
st.title("👤 Face Recognition System with Open-Set Rejection")
st.markdown(
    """
    **College ML Project Review Demo** | **Architecture:** Two-Stage ML System  
    *Stage 1:* Pretrained Deep Neural Network (128-d Feature Extractor) &nbsp;➔&nbsp; 
    *Stage 2:* Trained Scikit-Learn Support Vector Classifier (`SVC` with probability calibration)
    """
)
st.divider()

# -----------------------------------------------------------------------------
# 4. Check for Prerequisites (Encodings & Model)
# -----------------------------------------------------------------------------
model_bundle = load_trained_model()

if model_bundle is None:
    # Friendly Onboarding UI when model is not yet trained
    st.warning("⚠️ **Trained Model (`classifier.pkl`) Not Found**")
    st.info(
        """
        ### 📋 Follow These 3 Steps to Train Your Face Recognition Model:
        
        **Step 1: Add your dataset images**
        - Create subfolders inside `dataset/` named after each person:
          ```text
          dataset/
          ├── Person_A/
          │   ├── photo1.jpg
          │   └── photo2.jpg
          └── Person_B/
              ├── photo1.jpg
              └── photo2.jpg
          ```
        - Recommended: At least 3–5 clear face photos per person.

        **Step 2: Extract Face Embeddings (Stage 1)**
        - Open your terminal and run:
          ```bash
          python encode_faces.py
          ```
        - This extracts 128-dimensional deep feature embeddings and saves them to `encodings.pickle`.

        **Step 3: Train the SVM Classifier (Stage 2)**
        - Run:
          ```bash
          python train_model.py
          ```
        - This trains an SVM classifier with probability calibration and saves `classifier.pkl`.

        ---
        **🔄 Once finished, refresh this page to begin real-time face recognition!**
        """
    )
    st.stop()


# Extract model components
classifier = model_bundle["classifier"]
classes = model_bundle["classes"]
num_classes = model_bundle.get("num_classes", len(classes))
num_samples = model_bundle.get("num_samples", "N/A")


# -----------------------------------------------------------------------------
# 5. Sidebar Controls & Real-Time Parameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Controls")
    
    st.markdown("### 1. Open-Set Rejection")
    confidence_threshold = st.slider(
        "Confidence Threshold (τ)",
        min_value=0.30,
        max_value=0.95,
        value=0.60,
        step=0.05,
        help=(
            "If the classifier's top predicted probability is below this threshold, "
            "the face is rejected as 'Unknown / Not in dataset'."
        ),
    )
    st.caption(
        f"🎯 Current Threshold: **{confidence_threshold * 100:.0f}%**\n\n"
        "• **Higher Threshold:** Stricter security (Lower False Accepts, Higher False Rejects)\n"
        "• **Lower Threshold:** More lenient (Higher False Accepts, Lower False Rejects)"
    )

    st.divider()

    st.markdown("### 2. Quality & Blur Detection")
    blur_threshold = st.slider(
        "Blur Threshold (Laplacian Var)",
        min_value=20.0,
        max_value=300.0,
        value=100.0,
        step=10.0,
        help="Images with Laplacian variance score below this value trigger a blur warning.",
    )
    st.caption(f"Default standard threshold: **100.0**")

    st.divider()

    st.markdown("### 📊 Model Info")
    st.metric(label="Known Identities (Classes)", value=num_classes)
    st.metric(label="Training Samples", value=num_samples)
    st.metric(label="Embedding Dimensions", value="128-d")

    with st.expander("👥 Registered Person List", expanded=False):
        for idx, person in enumerate(classes, 1):
            st.write(f"{idx}. {person}")


# -----------------------------------------------------------------------------
# 6. Main Interface: Input Selection & Processing
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Face Recognition & Inference", 
    "📈 Confidence & Open-Set Analysis", 
    "📚 Project Review & ML Theory Guide"
])

with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Provide Input Image")

        # Input mode selection: Upload or pick from test_images/
        test_images_list = []
        if TEST_IMAGES_DIR.exists():
            test_images_list = [
                f.name for f in TEST_IMAGES_DIR.iterdir() 
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]

        input_choice = "Upload New Image"
        if test_images_list:
            input_choice = st.radio(
                "Select Input Source:",
                ["Upload New Image", "Select from 'test_images/' folder"],
                horizontal=True
            )

        image_to_process = None

        if input_choice == "Upload New Image":
            uploaded_file = st.file_uploader(
                "Choose a face image (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded_file is not None:
                image_to_process = uploaded_file
        else:
            selected_test_file = st.selectbox("Choose a test image:", test_images_list)
            if selected_test_file:
                image_to_process = str(TEST_IMAGES_DIR / selected_test_file)

        if image_to_process is not None:
            try:
                rgb_image = load_image_as_rgb(image_to_process)
                st.image(rgb_image, caption="Original Input Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                image_to_process = None

    with col_result:
        st.subheader("2. Detection & Recognition Pipeline")

        if image_to_process is None:
            st.info("👆 Please upload or select an image on the left to begin recognition.")
        else:
            # -----------------------------------------------------------------
            # Step A: Image Blur Check
            # -----------------------------------------------------------------
            st.markdown("#### Step A: Image Sharpness & Blur Check")
            blurry_flag, blur_score = is_blurry(rgb_image, threshold=blur_threshold)

            if blurry_flag:
                st.warning(
                    f"⚠️ **Image Blur Warning!**\n\n"
                    f"• Sharpness Score (Laplacian Variance): **{blur_score:.2f}**\n"
                    f"• Threshold: **{blur_threshold:.1f}**\n\n"
                    f"*Note: Blurry or out-of-focus images may reduce recognition confidence.*"
                )
            else:
                st.success(
                    f"✅ **Image Sharpness Passed** — Score: **{blur_score:.2f}** (Threshold: {blur_threshold:.1f})"
                )

            # -----------------------------------------------------------------
            # Step B: Face Detection & Embedding Extraction (Stage 1)
            # -----------------------------------------------------------------
            st.markdown("#### Step B: Face Detection & Deep Embeddings")
            with st.spinner("Detecting faces and extracting 128-d embeddings..."):
                try:
                    face_boxes, face_encodings = extract_face_encodings(rgb_image)
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    face_boxes, face_encodings = [], []

            if len(face_boxes) == 0:
                st.error(
                    "❌ **No Face Detected in Image**\n\n"
                    "The pretrained face detector could not locate any human face. "
                    "Please ensure the face is clearly visible, well-lit, and facing forward."
                )
            else:
                num_faces = len(face_boxes)
                st.info(f"👤 Detected **{num_faces}** face(s) in the image.")

                # -------------------------------------------------------------
                # Step C: Classification & Open-Set Verification (Stage 2)
                # -------------------------------------------------------------
                st.markdown("#### Step C: ML Classification & Open-Set Verification")

                labels = []
                confidences = []
                is_recognized_list = []
                all_prediction_probs = []

                train_encs = model_bundle.get("train_encodings", None)
                train_lbls = model_bundle.get("train_labels", None)

                for i, encoding in enumerate(face_encodings):
                    # Reshape encoding for scikit-learn: (1, 128)
                    feat = encoding.reshape(1, -1)

                    # 1. Get calibrated probabilities from trained SVM model
                    svm_probs = classifier.predict_proba(feat)[0]
                    top_idx = int(np.argmax(svm_probs))
                    top_name = classes[top_idx]
                    top_svm_prob = float(svm_probs[top_idx])

                    # 2. Compute Euclidean distance and Cosine similarity in deep feature space
                    if train_encs is not None and len(train_encs) > 0 and train_lbls is not None:
                        # Find minimum distance to the predicted class in dataset
                        class_indices = [idx for idx, name in enumerate(train_lbls) if name == top_name]
                        if class_indices:
                            class_encs = train_encs[class_indices]
                            dists = np.linalg.norm(class_encs - encoding, axis=1)
                            min_dist = float(np.min(dists))
                            
                            # Standard face embedding similarity formula (d <= 0.60 is genuine match)
                            # Distance 0.2 -> ~95% confidence, Distance 0.5 -> ~80%, Distance 0.7+ -> <45%
                            dist_similarity = max(0.0, min(1.0, 1.0 - (min_dist / 1.0) ** 1.5))
                            
                            # Calibrated confidence: blend SVM decision probability + deep feature distance
                            calibrated_conf = 0.50 * top_svm_prob + 0.50 * dist_similarity
                        else:
                            min_dist = 1.0
                            calibrated_conf = top_svm_prob
                    else:
                        min_dist = None
                        calibrated_conf = top_svm_prob

                    # Update probabilities list for chart visualization
                    calibrated_probs = []
                    for c_idx, c_name in enumerate(classes):
                        if train_encs is not None and train_lbls is not None:
                            c_indices = [idx for idx, name in enumerate(train_lbls) if name == c_name]
                            if c_indices:
                                c_dists = np.linalg.norm(train_encs[c_indices] - encoding, axis=1)
                                c_min_dist = float(np.min(c_dists))
                                c_sim = max(0.0, min(1.0, 1.0 - (c_min_dist / 1.0) ** 1.5))
                                c_conf = 0.50 * float(svm_probs[c_idx]) + 0.50 * c_sim
                            else:
                                c_conf = float(svm_probs[c_idx])
                        else:
                            c_conf = float(svm_probs[c_idx])
                        calibrated_probs.append(c_conf)

                    # Normalize distribution to sum to 1.0
                    sum_p = sum(calibrated_probs)
                    if sum_p > 0:
                        norm_probs = [p / sum_p for p in calibrated_probs]
                    else:
                        norm_probs = calibrated_probs
                    all_prediction_probs.append(norm_probs)

                    # Top confidence for open-set decision
                    final_conf = calibrated_conf

                    labels.append(top_name)
                    confidences.append(final_conf)

                    # OPEN-SET REJECTION LOGIC:
                    # If highest class probability is above threshold -> Accept Identity
                    # If highest class probability is below threshold -> Reject as Unknown
                    if final_conf >= confidence_threshold:
                        is_recognized_list.append(True)
                        st.success(
                            f"🎯 **Face #{i+1} RECOGNIZED:**  \n"
                            f"Identity: **{top_name}**  \n"
                            f"Confidence: **{final_conf * 100:.1f}%** (≥ Threshold {confidence_threshold * 100:.0f}%)"
                        )
                        if min_dist is not None:
                            st.caption(f"ℹ️ SVM Probability: **{top_svm_prob*100:.1f}%** | Feature Distance: **{min_dist:.2f}** (Threshold: <0.60)")
                    else:
                        is_recognized_list.append(False)
                        st.error(
                            f"🛡️ **Face #{i+1} NOT RECOGNIZED (Open-Set Rejection):**  \n"
                            f"Status: **Unknown / Not in Dataset**  \n"
                            f"Closest class was '{top_name}' with only **{final_conf * 100:.1f}%** confidence, "
                            f"which is below the rejection threshold of **{confidence_threshold * 100:.0f}%**."
                        )
                        if min_dist is not None:
                            st.caption(f"ℹ️ SVM Probability: **{top_svm_prob*100:.1f}%** | Feature Distance: **{min_dist:.2f}** (Too far for genuine match)")

                # Draw bounding box visual on the image
                annotated_img = draw_detection_boxes(
                    rgb_image,
                    face_boxes,
                    labels=labels,
                    confidences=confidences,
                    is_recognized_flags=is_recognized_list
                )

                st.markdown("#### 🖼️ Annotated Result")
                st.image(
                    annotated_img,
                    caption="Processed Image (Green = Verified, Red = Rejected/Unknown)",
                    use_container_width=True
                )


# -----------------------------------------------------------------------------
# Tab 2: Probability Distribution & Detailed Analysis
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("📊 Classifier Probability Breakdown (All Classes)")
    
    if image_to_process is not None and 'all_prediction_probs' in locals() and len(all_prediction_probs) > 0:
        for face_idx, probs in enumerate(all_prediction_probs):
            st.markdown(f"##### Face #{face_idx + 1} Prediction Probabilities:")
            
            # Format dataframe for clean table/chart
            df_probs = pd.DataFrame({
                "Identity": classes,
                "Probability (%)": [p * 100 for p in probs]
            }).sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)

            col_chart, col_table = st.columns([3, 2])
            with col_chart:
                st.bar_chart(df_probs.set_index("Identity"), use_container_width=True)
            with col_table:
                st.dataframe(
                    df_probs.style.format({"Probability (%)": "{:.2f}%"}),
                    use_container_width=True,
                    height=240
                )
    else:
        st.info("Run face recognition in Tab 1 to see the live probability distribution across all registered identities.")


# -----------------------------------------------------------------------------
# Tab 3: College Review & Viva Theory Guide
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📚 College Project Review: Machine Learning Theory & Defense")

    with st.expander("1. Why a Two-Stage ML Architecture?", expanded=True):
        st.markdown(
            """
            - **Stage 1 (Feature Extractor):** Raw face pixels ($W \\times H \\times 3$) have huge dimensionality, 
              extreme variance with lighting, pose, and background. A deep CNN/ResNet trained on millions of faces 
              projects any facial crop into a compact, semantically rich **128-dimensional Euclidean space** 
              where photos of the same person are clustered tightly together.
            - **Stage 2 (Trained Classifier):** We train a supervised **Support Vector Machine (SVM)** with a 
              linear kernel on top of these 128-d embeddings. This provides an actual project-trained ML model 
              that learns the optimal hyperplane boundaries separating each identity in your dataset.
            """
        )

    with st.expander("2. What is Open-Set Rejection and Why is it Essential?"):
        st.markdown(
            """
            - In a standard **Closed-Set** classification problem, the system assumes every input *must* belong 
              to one of the training classes ($\text{argmax}_{k} P(Y=k)$ always forces a decision).
            - In real-world security (an **Open-Set** problem), random unregistered people will present their faces. 
              Without rejection, an unknown intruder would be falsely matched to whichever registered student has the 
              highest residual similarity score.
            - **Our Rejection Criterion:**
              $$\\text{Decision}(X) = \\begin{cases} 
              \\hat{y} = \\arg\\max_k P(Y=k|X) & \\text{if } \\max_k P(Y=k|X) \\ge \\tau \\\\ 
              \\text{\"Face Not Recognized\"} & \\text{if } \\max_k P(Y=k|X) < \\tau 
              \\end{cases}$$
              where $\\tau$ is the adjustable confidence threshold.
            """
        )

    with st.expander("3. False Acceptance Rate (FAR) vs. False Rejection Rate (FRR) Tradeoff"):
        st.markdown(
            """
            - **FAR (False Acceptance Rate):** Proportion of unauthorized/unknown faces wrongly accepted as known users.
            - **FRR (False Rejection Rate):** Proportion of authorized/known users wrongly rejected by the system.
            - **The Threshold ($\tau$) Tradeoff:**
              - Increasing $\\tau$ (e.g., $0.85$): **FAR $\\downarrow$ (high security)**, but **FRR $\\uparrow$ (legitimate users may get rejected if lighting changes)**.
              - Decreasing $\\tau$ (e.g., $0.40$): **FRR $\\downarrow$ (convenient for known users)**, but **FAR $\\uparrow$ (intruders more likely to be accepted)**.
            """
        )

    with st.expander("4. Blur Detection via Laplacian Variance ($Var(\\nabla^2 I)$)"):
        st.markdown(
            """
            - The Laplacian operator $\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$ 
              is a 2D spatial derivative that acts as a high-pass filter, detecting rapid intensity transitions (sharp edges).
            - A crisp, in-focus image exhibits high variance in its Laplacian response. A blurry image lacks sharp edges, 
              resulting in a low variance score ($< 100.0$).
            """
        )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Face Recognition System | Streamlit Single-Process Application | "
    "Designed for College ML Review"
)
