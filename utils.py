"""
utils.py - Core utility functions for Face Recognition System.
Includes:
- Blur detection using Laplacian variance
- Face detection and 128-dimensional embedding extraction (face_recognition / DeepFace fallback)
- Image format conversion and bounding box annotation
"""

import cv2
import numpy as np
from PIL import Image
import io

# Try importing face_recognition; fallback gracefully if not yet compiled/installed
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    face_recognition = None
    FACE_REC_AVAILABLE = False

# Try importing deepface as secondary fallback
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False


def is_blurry(image, threshold: float = 100.0) -> tuple[bool, float]:
    """
    Detect if an image is blurry using the Variance of the Laplacian method.
    
    Mathematical intuition:
    The Laplacian operator highlights regions of rapid intensity change (edges).
    A sharp image will have high variance in edge responses (high Laplacian variance),
    whereas a blurry image will have low variance due to smoothed edges.

    Parameters:
        image: numpy array (RGB/BGR/Grayscale) or PIL Image
        threshold (float): Threshold below which the image is classified as blurry (default: 100.0)

    Returns:
        tuple[bool, float]: (is_blurry, variance_score)
    """
    # Convert PIL Image or bytes to numpy array
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    elif isinstance(image, (bytes, bytearray, io.BytesIO)):
        if isinstance(image, io.BytesIO):
            image = image.getvalue()
        file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        img_np = image.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Convert to grayscale if needed
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:  # RGB/BGR
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # Compute the Laplacian of the image and return the variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance_score = float(laplacian.var())
    
    is_blur = variance_score < threshold
    return is_blur, variance_score


def load_image_as_rgb(image_input) -> np.ndarray:
    """
    Load an image from various input types (file path, bytes, PIL Image, Streamlit UploadedFile)
    and return as a standard C-contiguous uint8 RGB numpy array.
    """
    if isinstance(image_input, str):
        # File path
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image from path: {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
    
    elif hasattr(image_input, "read") or isinstance(image_input, (bytes, bytearray, io.BytesIO)):
        # Streamlit UploadedFile or bytes
        if hasattr(image_input, "read"):
            image_bytes = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
        elif isinstance(image_input, io.BytesIO):
            image_bytes = image_input.getvalue()
        else:
            image_bytes = image_input

        pil_img = Image.open(io.BytesIO(image_bytes))
        rgb = np.array(pil_img.convert("RGB"))
    
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image_input.copy()
    else:
        raise TypeError(f"Cannot load image of type {type(image_input)}")
    
    # Ensure standard 8-bit contiguous array
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def extract_face_encodings(image_rgb: np.ndarray, model: str = "hog") -> tuple[list, list]:
    """
    Detect human faces in an RGB image and extract 128-dimensional deep embeddings.

    Parameters:
        image_rgb (np.ndarray): RGB image as a numpy array.
        model (str): Face detection model ('hog' for CPU fast, 'cnn' for GPU).

    Returns:
        tuple[list, list]:
            - face_locations: list of tuples (top, right, bottom, left)
            - face_encodings: list of 128-dimensional numpy vectors
    """
    # Ensure contiguous 8-bit RGB array
    img = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if FACE_REC_AVAILABLE:
        # 1. Detect face bounding box locations
        boxes = face_recognition.face_locations(img, model=model)
        if not boxes:
            return [], []
        
        # 2. Extract 128-dimensional embeddings for each detected face
        encodings = face_recognition.face_encodings(img, known_face_locations=boxes)
        return boxes, encodings
    
    elif DEEPFACE_AVAILABLE:
        # Fallback to DeepFace with ArcFace/Facenet backend if dlib is not available
        try:
            results = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="opencv"
            )
            boxes = []
            encodings = []
            for item in results:
                area = item.get("facial_area", {})
                x, y, w, h = area.get("x", 0), area.get("y", 0), area.get("w", 0), area.get("h", 0)
                boxes.append((y, x + w, y + h, x))
                encodings.append(np.array(item.get("embedding", [])))
            return boxes, encodings
        except Exception:
            return [], []
    else:
        raise ImportError(
            "Neither 'face_recognition' (dlib) nor 'deepface' is installed. "
            "Please run: pip install face_recognition (or pip install deepface)"
        )


def draw_detection_boxes(
    image_rgb: np.ndarray,
    face_locations: list,
    labels: list[str] = None,
    confidences: list[float] = None,
    is_recognized_flags: list[bool] = None
) -> np.ndarray:
    """
    Draw professional bounding boxes and identity badges on the RGB image.

    Colors:
    - Green (RGB: 34, 197, 94): Recognized (Above threshold)
    - Red (RGB: 239, 68, 68): Unrecognized / Rejected (Below threshold)
    - Blue (RGB: 59, 130, 246): Detected (No classification)
    """
    annotated = np.ascontiguousarray(image_rgb.copy(), dtype=np.uint8)
    h, w, _ = annotated.shape

    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Determine status and color
        is_rec = is_recognized_flags[i] if is_recognized_flags and i < len(is_recognized_flags) else None
        label = labels[i] if labels and i < len(labels) else None
        conf = confidences[i] if confidences and i < len(confidences) else None

        if is_rec is True:
            box_color = (34, 197, 94)      # Emerald Green
            header_text = f"{label} ({conf*100:.1f}%)" if conf is not None else label
        elif is_rec is False:
            box_color = (239, 68, 68)      # Coral Red
            header_text = f"Unknown ({conf*100:.1f}%)" if conf is not None else "Unknown"
        else:
            box_color = (59, 130, 246)     # Sky Blue
            header_text = label if label else "Face Detected"

        # Draw outer rectangle
        cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)

        # Draw corner accents for a modern sleek look
        corner_len = min(20, (right - left) // 4, (bottom - top) // 4)
        thickness = 4
        # Top-left
        cv2.line(annotated, (left, top), (left + corner_len, top), box_color, thickness)
        cv2.line(annotated, (left, top), (left, top + corner_len), box_color, thickness)
        # Top-right
        cv2.line(annotated, (right, top), (right - corner_len, top), box_color, thickness)
        cv2.line(annotated, (right, top), (right, top + corner_len), box_color, thickness)
        # Bottom-left
        cv2.line(annotated, (left, bottom), (left + corner_len, bottom), box_color, thickness)
        cv2.line(annotated, (left, bottom), (left, bottom - corner_len), box_color, thickness)
        # Bottom-right
        cv2.line(annotated, (right, bottom), (right - corner_len, bottom), box_color, thickness)
        cv2.line(annotated, (right, bottom), (right, bottom - corner_len), box_color, thickness)

        # Draw label background badge
        if header_text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(header_text, font, font_scale, font_thickness)

            badge_top = max(0, top - text_h - 10)
            badge_bottom = top
            badge_right = min(w, left + text_w + 14)

            # Badge background
            cv2.rectangle(annotated, (left, badge_top), (badge_right, badge_bottom), box_color, cv2.FILLED)
            # Badge text (White)
            cv2.putText(
                annotated,
                header_text,
                (left + 7, badge_bottom - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA
            )

    return annotated
