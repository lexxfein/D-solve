import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


# ============================================================
# MediaPipe model
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)


# ============================================================
# Detect face
# ============================================================

def detect_face_landmarks(image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1
    )

    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# Extract mouth from full face
# ============================================================

def extract_mouth_from_face(image, landmarks):

    height, width = image.shape[:2]

    # Outer mouth landmarks
    outer_indices = [
        61, 185, 40, 39, 37, 0,
        267, 269, 270, 409, 291,
        375, 321, 405, 314, 17,
        84, 181, 91
    ]

    points = []

    for index in outer_indices:

        point = landmarks[index]

        x = int(point.x * width)
        y = int(point.y * height)

        points.append([x, y])

    points = np.array(
        points,
        dtype=np.int32
    )

    # Bounding box
    x_min = max(
        np.min(points[:, 0]) - 15,
        0
    )

    x_max = min(
        np.max(points[:, 0]) + 15,
        width
    )

    y_min = max(
        np.min(points[:, 1]) - 15,
        0
    )

    y_max = min(
        np.max(points[:, 1]) + 15,
        height
    )

    mouth = image[
        y_min:y_max,
        x_min:x_max
    ].copy()

    # --------------------------------------------------------
    # INNER mouth opening
    # --------------------------------------------------------

    inner_indices = [
        78, 191, 80, 81, 82,
        13,
        312, 311, 310, 415, 308,
        324, 318, 402, 317,
        14,
        87, 178, 88, 95
    ]

    inner_points = []

    for index in inner_indices:

        point = landmarks[index]

        x = int(point.x * width) - x_min
        y = int(point.y * height) - y_min

        inner_points.append([x, y])

    inner_points = np.array(
        inner_points,
        dtype=np.int32
    )

    mouth_region = np.zeros(
        mouth.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mouth_region,
        [inner_points],
        255
    )

    return mouth, mouth_region


# ============================================================
# Mouth-only mode
# ============================================================

def use_mouth_only_image(image):

    height, width = image.shape[:2]

    region = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # Leave a small border out
    x1 = int(width * 0.03)
    x2 = int(width * 0.97)

    y1 = int(height * 0.03)
    y2 = int(height * 0.97)

    cv2.rectangle(
        region,
        (x1, y1),
        (x2, y2),
        255,
        -1
    )

    return image.copy(), region


# ============================================================
# Adaptive tooth mask
# ============================================================

def create_tooth_mask(mouth, mouth_region):

    lab = cv2.cvtColor(
        mouth,
        cv2.COLOR_BGR2LAB
    )

    hsv = cv2.cvtColor(
        mouth,
        cv2.COLOR_BGR2HSV
    )

    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    # --------------------------------------------------------
    # Only examine pixels inside mouth
    # --------------------------------------------------------

    valid = L[
        mouth_region > 0
    ]

    if len(valid) == 0:

        return np.zeros(
            L.shape,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # Adaptive brightness
    # --------------------------------------------------------

    # Lower than before so darker corner teeth
    # have a chance to be detected.

    threshold = np.percentile(
        valid,
        55
    )

    threshold = max(
        115,
        min(
            threshold,
            175
        )
    )

    bright = cv2.inRange(
        L,
        threshold,
        255
    )

    # --------------------------------------------------------
    # Teeth should not be strongly saturated
    # --------------------------------------------------------

    low_saturation = cv2.inRange(
        S,
        0,
        145
    )

    # --------------------------------------------------------
    # Avoid strongly red/pink pixels
    # --------------------------------------------------------

    not_red = cv2.inRange(
        A,
        95,
        155
    )

    # --------------------------------------------------------
    # Teeth can be white or yellowish.
    # Use a broad B range instead of excluding
    # darker/yellower corner teeth.
    # --------------------------------------------------------

    tooth_colour = cv2.inRange(
        B,
        115,
        205
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    tooth_mask = cv2.bitwise_and(
        bright,
        low_saturation
    )

    tooth_mask = cv2.bitwise_and(
        tooth_mask,
        not_red
    )

    tooth_mask = cv2.bitwise_and(
        tooth_mask,
        tooth_colour
    )

    # Restrict to inner mouth
    tooth_mask = cv2.bitwise_and(
        tooth_mask,
        mouth_region
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    tooth_mask = cv2.morphologyEx(
        tooth_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    tooth_mask = cv2.morphologyEx(
        tooth_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # Remove very small isolated regions
    # --------------------------------------------------------

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            tooth_mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(
        tooth_mask
    )

    image_area = (
        mouth.shape[0] *
        mouth.shape[1]
    )

    minimum_area = max(
        5,
        int(image_area * 0.00015)
    )

    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:

            cleaned[
                labels == i
            ] = 255

    # --------------------------------------------------------
# Protect against mask leakage
# --------------------------------------------------------

# Shrink the mask slightly so whitening does not
# spill onto lips, gums, or skin.

    protect_kernel = np.ones(
        (3, 3),
        np.uint8
    )

    protected = cv2.erode(
        cleaned,
        protect_kernel,
        iterations=1
    )

    return protected


# ============================================================
# Apply controlled whitening
# ============================================================

def apply_whitening(
    mouth,
    tooth_mask,
    intensity
):

    # --------------------------------------------------------
    # NON-LINEAR intensity
    #
    # Prevents 100 from becoming ridiculously strong.
    # --------------------------------------------------------

    user_strength = intensity / 100.0

# Keep high intensity strong, but prevent
# unrealistic over-processing.

    strength = (
        user_strength ** 0.9
    )

    # Maximum actual transformation
    # is intentionally limited.

    brightness_amount = (
        28 * strength
    )

    yellow_amount = (
        10 * strength
    )

    # --------------------------------------------------------
    # LAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        mouth,
        cv2.COLOR_BGR2LAB
    )

    L, A, B = cv2.split(lab)

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    L_float = L.astype(
        np.float32
    )

    L_float += brightness_amount

    L_float = np.clip(
        L_float,
        0,
        255
    )

    L_new = L_float.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Reduce yellow
    # --------------------------------------------------------

    B_float = B.astype(
        np.float32
    )

    B_float -= yellow_amount

    B_float = np.clip(
        B_float,
        0,
        255
    )

    B_new = B_float.astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    whitened_lab = cv2.merge(
        [L_new, A, B_new]
    )

    whitened = cv2.cvtColor(
        whitened_lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # Smooth mask
    # --------------------------------------------------------

    # --------------------------------------------------------
# Create a controlled soft mask
# --------------------------------------------------------

    soft_mask = cv2.GaussianBlur(
        tooth_mask,
        (7, 7),
        0
    )

    soft_mask = (
        soft_mask.astype(
            np.float32
        ) / 255.0
    )

    # Never apply the transformation at 100%
    # strength to every selected pixel.

    soft_mask *= 0.95

    soft_mask = soft_mask[
        :, :, np.newaxis
    ]

    # --------------------------------------------------------
    # Blend
    # --------------------------------------------------------

    result = (
        mouth * (1 - soft_mask)
        +
        whitened * soft_mask
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return result


# ============================================================
# MAIN FUNCTION
# ============================================================

def whiten_smile(
    image,
    intensity=50
):

    if image is None:

        raise ValueError(
            "Input image is empty"
        )

    intensity = max(
        0,
        min(
            100,
            int(intensity)
        )
    )

    # --------------------------------------------------------
    # Try face detection
    # --------------------------------------------------------

    landmarks = detect_face_landmarks(
        image
    )

    # --------------------------------------------------------
    # Full face
    # --------------------------------------------------------

    if landmarks is not None:

        mouth, mouth_region = (
            extract_mouth_from_face(
                image,
                landmarks
            )
        )

        full_image = True

    # --------------------------------------------------------
    # Mouth-only
    # --------------------------------------------------------

    else:

        mouth, mouth_region = (
            use_mouth_only_image(
                image
            )
        )

        full_image = False

    # --------------------------------------------------------
    # Tooth segmentation
    # --------------------------------------------------------

    tooth_mask = create_tooth_mask(
        mouth,
        mouth_region
    )

    # --------------------------------------------------------
    # Whitening
    # --------------------------------------------------------

    processed_mouth = apply_whitening(
        mouth,
        tooth_mask,
        intensity
    )

    # --------------------------------------------------------
    # Put mouth back into full image
    # --------------------------------------------------------

    if full_image:

        height, width = image.shape[:2]

        mouth_indices = [
            61, 185, 40, 39, 37, 0,
            267, 269, 270, 409, 291,
            375, 321, 405, 314, 17,
            84, 181, 91
        ]

        points = []

        for index in mouth_indices:

            point = landmarks[index]

            x = int(point.x * width)
            y = int(point.y * height)

            points.append([x, y])

        points = np.array(
            points,
            dtype=np.int32
        )

        x_min = max(
            np.min(points[:, 0]) - 15,
            0
        )

        x_max = min(
            np.max(points[:, 0]) + 15,
            width
        )

        y_min = max(
            np.min(points[:, 1]) - 15,
            0
        )

        y_max = min(
            np.max(points[:, 1]) + 15,
            height
        )

        output = image.copy()

        output[
            y_min:y_max,
            x_min:x_max
        ] = processed_mouth

        return output

    return processed_mouth