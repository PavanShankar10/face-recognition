"""
train_model.py - Model Training Stage (Stage 2)
-----------------------------------------------
Loads the extracted face embeddings from 'encodings.pickle', trains a
Support Vector Machine (SVM) classifier with probability calibration
(SVC with probability=True), evaluates performance on a test split,
and saves the trained model bundle to 'classifier.pkl'.

Usage:
    python train_model.py
"""

import os
import sys
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ENCODINGS_FILE = Path("encodings.pickle")
MODEL_FILE = Path("classifier.pkl")


def main():
    print("=" * 65)
    print("  STAGE 2: TRAIN MACHINE LEARNING CLASSIFIER (SVM with Probabilities)")
    print("=" * 65)

    # 1. Check if encodings exist
    if not ENCODINGS_FILE.exists():
        print(f"\n[ERROR] Encodings file '{ENCODINGS_FILE}' not found.")
        print("Please run Stage 1 first to extract face embeddings:")
        print("    python encode_faces.py\n")
        sys.exit(1)

    # 2. Load encodings
    print(f"\n[INFO] Loading face embeddings from '{ENCODINGS_FILE}'...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    raw_encodings = data.get("encodings", [])
    raw_names = data.get("names", [])

    if len(raw_encodings) == 0 or len(raw_names) == 0:
        print("[ERROR] Encodings file is empty. Please re-run 'encode_faces.py' with valid images.")
        sys.exit(1)

    X = np.array(raw_encodings)
    y = np.array(raw_names)

    total_samples = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    num_classes = len(unique_classes)

    print(f"[INFO] Loaded {total_samples} samples across {num_classes} distinct identities:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"    * {cls:<20}: {count} sample(s)")

    if num_classes < 2:
        print("\n[WARNING] Found fewer than 2 distinct people in the dataset.")
        print("A classification model needs at least 2 distinct classes (people) to distinguish between them.")
        print("Please add at least one more person's folder in 'dataset/' and re-run 'encode_faces.py'.")

    # 3. Encode string names to integer class IDs
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Check class distribution for train/test split
    min_count = min(class_counts)
    can_split = total_samples >= 4 and min_count >= 2 and num_classes >= 2

    if not can_split:
        print("\n[WARNING] Dataset is small or some identities have fewer than 2 images.")
        print("           Recommended: At least 3-5 images per person for train/test evaluation.")
        print("           Training classifier on 100% of available samples...")
        X_train, y_train = X, y_encoded
        X_test, y_test = None, None
    else:
        # Perform 80/20 train/test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=0.20,
                random_state=42,
                stratify=y_encoded
            )
            print(f"\n[INFO] Dataset split: {len(y_train)} training samples (80%), {len(y_test)} testing samples (20%).")
        except ValueError:
            # Fallback if stratify fails due to single-instance classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=0.20,
                random_state=42
            )
            print(f"\n[INFO] Dataset split (non-stratified): {len(y_train)} train, {len(y_test)} test.")

    # 5. Train Support Vector Machine (SVM) Classifier
    # High-dimensional normalized face embeddings (128-d) benefit from C=5.0 for clear margin separation
    print("\n[INFO] Training Support Vector Classifier (SVC)...")
    print("       Hyperparameters: kernel='linear', C=5.0, probability=True")

    clf = SVC(C=5.0, kernel="linear", probability=True, random_state=42)
    clf.fit(X_train, y_train)
    print("[+] Model training completed successfully.")

    # 6. Model Evaluation
    print("\n" + "=" * 65)
    print("                    MODEL EVALUATION REPORT")
    print("=" * 65)

    if X_test is not None and len(y_test) > 0:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Test Split Overall Accuracy: {acc * 100:.2f}%\n")
        print("  Detailed Classification Report (Precision, Recall, F1-Score):")
        
        test_unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [le.classes_[idx] for idx in test_unique_classes]
        
        report = classification_report(
            y_test,
            y_pred,
            labels=test_unique_classes,
            target_names=target_names,
            zero_division=0
        )
        print(report)

        # Train final production model on 100% of data to maximize accuracy
        print("[INFO] Re-fitting final production model on 100% of dataset samples...")
        clf.fit(X, y_encoded)
    else:
        train_pred = clf.predict(X_train)
        acc = accuracy_score(y_train, train_pred)
        print(f"  Training Accuracy: {acc * 100:.2f}% (Evaluated on all available samples)")
        print("  Note: Add >= 3 images per person to see separate train/test evaluation split.")

    print("=" * 65)

    # 7. Save model bundle
    # Bundle contains the trained model, label encoder, class names, dataset embeddings, and metadata
    model_bundle = {
        "classifier": clf,
        "label_encoder": le,
        "classes": list(le.classes_),
        "num_classes": num_classes,
        "num_samples": total_samples,
        "train_encodings": X,
        "train_labels": y,
        "embedding_dim": X.shape[1] if len(X.shape) > 1 else 128
    }

    print(f"\n[INFO] Serializing model bundle to '{MODEL_FILE}'...")
    joblib.dump(model_bundle, MODEL_FILE)

    print(f"[SUCCESS] Trained model saved to '{MODEL_FILE.resolve()}'")
    print("Next step: Launch the Streamlit application:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
