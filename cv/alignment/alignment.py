import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)


def detect_face_landmarks(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


def get_mouth_region(image, landmarks):
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

    points = np.array(points, dtype=np.int32)

    x_min = max(np.min(points[:, 0]) - 15, 0)
    x_max = min(np.max(points[:, 0]) + 15, width)

    y_min = max(np.min(points[:, 1]) - 15, 0)
    y_max = min(np.max(points[:, 1]) + 15, height)

    mouth = image[y_min:y_max, x_min:x_max].copy()

    # INNER mouth landmarks
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

    inner_mask = np.zeros(
        mouth.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        inner_mask,
        [inner_points],
        255
    )

    return (
        mouth,
        inner_mask,
        x_min,
        y_min,
        x_max,
        y_max
    )


def create_tooth_mask(mouth, inner_mask):

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

    # Only examine pixels INSIDE the inner mouth.
    valid_pixels = L[inner_mask > 0]

    if len(valid_pixels) == 0:
        return np.zeros_like(L)

    # Adaptive brightness threshold.
    brightness_threshold = np.percentile(
        valid_pixels,
        55
    )

    brightness_threshold = max(
        125,
        min(brightness_threshold, 190)
    )

    bright = cv2.inRange(
        L,
        brightness_threshold,
        255
    )

    # Teeth are generally less saturated than lips/gums.
    low_saturation = cv2.inRange(
        S,
        0,
        145
    )

    # Avoid strongly red/pink tissue.
    neutral_colour = cv2.inRange(
        A,
        90,
        155
    )

    # Combine colour conditions.
    mask = cv2.bitwise_and(
        bright,
        low_saturation
    )

    mask = cv2.bitwise_and(
        mask,
        neutral_colour
    )

    # CRITICAL:
    # Only allow pixels inside the mouth.
    mask = cv2.bitwise_and(
        mask,
        inner_mask
    )

    # Remove tiny noise.
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # Keep reasonably sized tooth regions.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    mouth_area = mouth.shape[0] * mouth.shape[1]

    minimum_area = max(
        5,
        int(mouth_area * 0.0001)
    )

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= minimum_area:
            cleaned[labels == i] = 255

    # Protect edges from bleeding.
    cleaned = cv2.erode(
        cleaned,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    return cleaned


def straighten_teeth(mouth, tooth_mask, intensity):

    intensity = max(
        0,
        min(100, int(intensity))
    )

    if intensity == 0:
        return mouth.copy()

    ys, xs = np.where(
        tooth_mask > 0
    )

    if len(xs) < 20:
        print("Not enough tooth pixels detected.")
        return mouth.copy()

    height, width = mouth.shape[:2]

    # Find connected tooth regions.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            tooth_mask,
            connectivity=8
        )
    )

    result = mouth.copy()

    strength = intensity / 100.0

    # Maximum movement in pixels.
    max_shift = 12 * strength

    # Get tooth components.
    teeth = []

    for i in range(1, num_labels):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < 20:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        cx, cy = centroids[i]

        teeth.append({
            "label": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area
        })

    if len(teeth) == 0:
        return mouth.copy()

    # Sort teeth from left to right.
    teeth.sort(
        key=lambda tooth: tooth["cx"]
    )

    print(
        "Detected tooth regions:",
        len(teeth)
    )

    # Calculate average tooth-row height.
    average_y = np.mean(
        [tooth["cy"] for tooth in teeth]
    )

    # Move each tooth toward a smoother row.
    for tooth in teeth:

        cx = tooth["cx"]
        cy = tooth["cy"]

        # Horizontal position from 0 to 1.
        normalized_x = (
            cx / max(width - 1, 1)
        )

        # Target curve.
        target_curve = (
            np.sin(normalized_x * np.pi)
        )

        target_y = (
            average_y
            - max_shift * target_curve
        )

        shift_y = target_y - cy

        # Limit movement.
        shift_y = np.clip(
            shift_y,
            -max_shift,
            max_shift
        )

        if abs(shift_y) < 1:
            continue

        x1 = max(
            tooth["x"] - 4,
            0
        )

        y1 = max(
            tooth["y"] - 4,
            0
        )

        x2 = min(
            tooth["x"]
            + tooth["w"]
            + 4,
            width
        )

        y2 = min(
            tooth["y"]
            + tooth["h"]
            + 4,
            height
        )

        patch = mouth[
            y1:y2,
            x1:x2
        ].copy()

        patch_mask = (
            labels[
                y1:y2,
                x1:x2
            ] == tooth["label"]
        ).astype(np.uint8) * 255

        # Move the individual tooth.
        matrix = np.float32([
            [1, 0, 0],
            [0, 1, shift_y]
        ])

        moved_patch = cv2.warpAffine(
            patch,
            matrix,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        moved_mask = cv2.warpAffine(
            patch_mask,
            matrix,
            (patch_mask.shape[1], patch_mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        moved_mask = cv2.GaussianBlur(
            moved_mask,
            (5, 5),
            0
        )

        alpha = (
            moved_mask.astype(np.float32)
            / 255.0
        )

        alpha = alpha[:, :, np.newaxis]

        original_area = result[
            y1:y2,
            x1:x2
        ].astype(np.float32)

        moved_area = moved_patch.astype(
            np.float32
        )

        blended = (
            original_area * (1 - alpha)
            + moved_area * alpha
        )

        result[
            y1:y2,
            x1:x2
        ] = np.clip(
            blended,
            0,
            255
        ).astype(np.uint8)

    return result


def align_smile(image, intensity=50):

    if image is None:
        raise ValueError(
            "Input image is empty"
        )

    intensity = max(
        0,
        min(100, int(intensity))
    )

    landmarks = detect_face_landmarks(
        image
    )

    if landmarks is None:
        print("No face detected.")
        return image.copy()

    (
        mouth,
        inner_mask,
        x_min,
        y_min,
        x_max,
        y_max
    ) = get_mouth_region(
        image,
        landmarks
    )

    tooth_mask = create_tooth_mask(
        mouth,
        inner_mask
    )

    processed_mouth = straighten_teeth(
        mouth,
        tooth_mask,
        intensity
    )

    output = image.copy()

    output[
        y_min:y_max,
        x_min:x_max
    ] = processed_mouth

    return output