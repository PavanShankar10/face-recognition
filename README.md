# Face Recognition System with Open-Set Rejection & Blur Detection

A complete, production-ready Machine Learning system built with a **Two-Stage ML Architecture** and an interactive **Streamlit** user interface.

---

## 🏛️ Architecture Overview

The system operates strictly as a self-contained Streamlit application (Streamlit acts as both frontend and inference engine with no external API servers).

```
+-------------------------------------------------------------------------+
|                              INPUT IMAGE                                |
+-------------------------------------------------------------------------+
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │        Step 1: Blur Detection          │
                │    Laplacian Variance: Var(∇² I)       │
                └───────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │      Step 2: Deep Face Detection       │
                │     & 128-d Embedding Extraction       │
                │       (Pretrained CNN / dlib)          │
                └───────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │   Step 3: Trained ML Classification    │
                │    Scikit-Learn Linear SVM (SVC)       │
                │     with Probability Calibration       │
                └───────────────────┬────────────────────┘
                                     │
                                     ▼
                 /─────────────────────────────────────\
                <  Top Probability P(Y=k|X) ≥ Threshold? >
                 \─────────────────────────────────────/
                         │                     │
                    YES  │                     │  NO
                         ▼                     ▼
              ┌─────────────────────┐   ┌─────────────────────┐
              │     RECOGNIZED      │   │ OPEN-SET REJECTION  │
              │  Show Name & Score  │   │ "Face Not Recognized│
              │   (Green Bounding)  │   │  Not in Dataset"    │
              └─────────────────────┘   └─────────────────────┘
```

---

## 📁 Project Folder Structure

```text
face_recognition_project/
├── dataset/                  # Place your dataset images here (one subfolder per person)
│   ├── Alice/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Bob/
│       ├── photo1.jpg
│       └── photo2.jpg
├── test_images/              # Place test images here for quick evaluation
│   ├── test_alice.jpg
│   └── unknown_person.jpg
├── encode_faces.py           # Stage 1: Extracts 128-d embeddings -> encodings.pickle
├── train_model.py            # Stage 2: Trains SVM classifier -> classifier.pkl
├── app.py                    # Streamlit Application (UI + Inference Pipeline)
├── utils.py                  # Blur detection & face extraction helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation & ML theory guide
```

---

## 🚀 Setup & Installation Instructions

### 1. Create and Activate Virtual Environment (Recommended)

```bash
# Open terminal in project root
python -m venv venv

# Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows Users (`dlib` / `face_recognition` installation):**  
> `face_recognition` relies on `dlib`. If you encounter build issues during `pip install face_recognition`:
> 1. Install CMake: `pip install cmake`
> 2. Ensure Visual Studio C++ Build Tools are installed, or install a pre-built wheel:
>    `pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl` (adjust for your Python version)
> 3. Alternatively, `utils.py` contains automatic fallback compatibility for `deepface` (`pip install deepface`).

---

## 🔄 Step-by-Step Pipeline Execution

### Step 1: Add Your Dataset Images
1. Create a subfolder inside `dataset/` for each individual.
2. Place at least **3 to 5 clear photos** per person in their respective folder.
   - Example:
     - `dataset/Elon_Musk/img1.jpg`, `dataset/Elon_Musk/img2.jpg`
     - `dataset/Sam_Altman/img1.jpg`, `dataset/Sam_Altman/img2.jpg`

### Step 2: Extract Face Embeddings (Stage 1)
```bash
python encode_faces.py
```
- Scans `dataset/`, detects faces, extracts 128-d embeddings, and saves `encodings.pickle`.

### Step 3: Train the Machine Learning Classifier (Stage 2)
```bash
python train_model.py
```
- Trains a scikit-learn `SVC(kernel='linear', probability=True)` classifier on the embeddings.
- Evaluates train/test split performance and saves the model bundle to `classifier.pkl`.

### Step 4: Launch the Streamlit Application
```bash
streamlit run app.py
```
- Opens the interactive web UI in your browser at `http://localhost:8501`.

---

## 🎓 College ML Project Review & Viva Theory Guide

### 1. Why a Two-Stage ML Pipeline?
- **Stage 1 (Feature Extractor - Deep Representation):** Raw image pixels ($1000 \times 1000 \times 3$) have huge variance due to lighting, background, and head pose. We use a pretrained deep ResNet model as a fixed feature extractor to transform each face crop into a compact, robust **128-dimensional embedding vector**. In this embedding space, Euclidean distance directly corresponds to facial similarity.
- **Stage 2 (Trained Classifier - Project ML Model):** We train a supervised **Support Vector Machine (SVM)** with a linear kernel on top of the 128-d embeddings. The SVM learns optimal maximum-margin decision hyperplanes separating each individual identity.

### 2. What is Open-Set Rejection and Why is Standard Classification Insufficient?
- **Closed-Set Assumption:** Standard classifiers assume any test input *must* belong to one of the training classes ($\arg\max_k P(Y=k|X)$ always picks a winner).
- **Open-Set Reality:** In a real-world security or attendance system, unregistered people (intruders/visitors) will appear. Without rejection, the system would misidentify an unknown person as whichever registered student happens to have the highest residual similarity.
- **Our Threshold Mechanism:**
  $$\text{Decision}(X) = \begin{cases} 
  \hat{y} = \arg\max_k P(Y=k|X) & \text{if } \max_k P(Y=k|X) \ge \tau \\ 
  \text{"Face Not Recognized"} & \text{if } \max_k P(Y=k|X) < \tau 
  \end{cases}$$
  where $\tau$ is the adjustable confidence threshold (slider in the Streamlit sidebar).

### 3. Understanding FAR vs. FRR Tradeoff
- **FAR (False Acceptance Rate):** Rate at which unauthorized/unknown faces are wrongly accepted as known users.
- **FRR (False Rejection Rate):** Rate at which authorized/known users are wrongly rejected by the system.
- **Slider Impact:**
  - **Higher Threshold ($\tau \ge 0.80$):** High security. FAR drops close to 0%, but FRR increases (legitimate users might get rejected under poor lighting).
  - **Lower Threshold ($\tau \le 0.45$):** User convenience. FRR drops, but FAR increases (intruders may be falsely accepted).

### 4. Evaluation Metrics from `train_model.py`
- **Accuracy:** Overall proportion of correct predictions: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all predictions made for Person A, how many were actually Person A?
- **Recall:** $\frac{TP}{TP + FN}$ — Out of all actual images of Person A, how many did the model find?
- **F1-Score:** Harmonic mean of Precision and Recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

### 5. Blur Detection via Laplacian Variance
- We compute the variance of the 2D Laplacian operator:
  $$\text{Score} = \text{Var}(\nabla^2 I) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
- Sharp images have sharp edge transitions $\rightarrow$ High Laplacian variance ($\ge 100$).
- Blurry images have smoothed transitions $\rightarrow$ Low Laplacian variance ($< 100$).
