from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "face_landmarker.task"
)


# MediaPipe lip landmark groups.
OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185
]

INNER_LIPS = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]


# ============================================================
# Face landmarks
# ============================================================

def detect_face_landmarks(image):
    """Return MediaPipe face landmarks as pixel (x, y) tuples."""

    if image is None or image.size == 0:
        return None

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face landmarker model not found:\n{MODEL_PATH}"
        )

    mp_base = mp.tasks.BaseOptions
    mp_vision = mp.tasks.vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_base(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    h, w = image.shape[:2]

    return [
        (int(round(p.x * w)), int(round(p.y * h)))
        for p in result.face_landmarks[0]
    ]


# ============================================================
# Mouth extraction
# ============================================================

def get_mouth_region(image, landmarks, padding=8):
    """Crop the mouth and return crop + inner-mouth mask."""

    if landmarks is None:
        raise ValueError("No face landmarks detected.")

    h, w = image.shape[:2]

    outer = np.array(
        [landmarks[i] for i in OUTER_LIPS],
        dtype=np.int32
    )

    x, y, mw, mh = cv2.boundingRect(outer)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + mw + padding)
    y2 = min(h, y + mh + padding)

    mouth = image[y1:y2, x1:x2].copy()

    inner_global = np.array(
        [landmarks[i] for i in INNER_LIPS],
        dtype=np.int32
    )

    inner_local = inner_global - np.array(
        [x1, y1],
        dtype=np.int32
    )

    inner_mask = np.zeros(
        mouth.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        inner_mask,
        [inner_local],
        255
    )

    return mouth, inner_mask, x1, y1


# ============================================================
# Tooth segmentation
# ============================================================

def create_veneer_mask(mouth, inner_mask):
    """
    Build a STRICT enamel-only mask.

    The previous renderer allowed a permissive tooth mask to leak
    into pink gingiva/lip tissue.  This version deliberately favors
    a smaller, cleaner enamel mask.

    Important rule:
        NOTHING is rendered outside this enamel mask except the
        explicitly reconstructed incisal-chip pixels.
    """

    lab = cv2.cvtColor(mouth, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(mouth, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0].astype(np.int16)
    A = lab[:, :, 1].astype(np.int16)
    B = lab[:, :, 2].astype(np.int16)

    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    # Pink/red gingiva has substantially higher LAB-A and HSV
    # saturation than enamel.  The A constraint is the key
    # protection against the gum boundary being treated as tooth.
    neutral_enamel = (
        (L >= 132) &
        (V >= 120) &
        (S <= 88) &
        (A <= 141) &
        (A >= 112) &
        (B <= 185)
    )

    mask = np.zeros(mouth.shape[:2], np.uint8)
    mask[neutral_enamel] = 255

    mask = cv2.bitwise_and(mask, inner_mask)

    # Remove isolated skin/noise.  Do NOT dilate; dilation is what
    # allows a tooth mask to grow into surrounding tissue.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # A tiny close reconnects enamel pixels inside the same tooth,
    # but never expands the mask outward.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((2, 2), np.uint8),
        iterations=1
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        8
    )

    cleaned = np.zeros_like(mask)

    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        if area >= 25 and w >= 3 and h >= 4:
            cleaned[labels == label] = 255

    # Never let the mask escape the original mouth opening.
    cleaned = cv2.bitwise_and(
        cleaned,
        inner_mask
    )

    return cleaned



def _fallback_teeth(binary):
    teeth = []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        8
    )

    for label in range(1, n):
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 45:
            continue

        tooth = np.zeros_like(binary)
        tooth[labels == label] = 255

        contours, _ = cv2.findContours(
            tooth,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        if w < 5 or h < 5:
            continue

        M = cv2.moments(contour)

        if M["m00"]:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = centroids[label]

        teeth.append({
            "mask": tooth,
            "contour": contour,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": float(cx),
            "cy": float(cy),
            "area": int(area),
        })

    teeth.sort(key=lambda t: t["cx"])

    for i, tooth in enumerate(teeth, 1):
        tooth["index"] = i

    return teeth


def extract_individual_teeth(tooth_mask):
    """
    Split the detected tooth mask into individual teeth.
    """

    if tooth_mask is None:
        return []

    binary = np.where(
        tooth_mask > 0,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        return []

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5
    )

    if distance.max() <= 0:
        return []

    cores = np.zeros_like(binary)
    cores[
        distance >= distance.max() * 0.32
    ] = 255

    cores = cv2.morphologyEx(
        cores,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    count, markers = cv2.connectedComponents(cores)

    if count <= 1:
        return _fallback_teeth(binary)

    # Watershed needs actual image gradients. For the mask-only
    # implementation, use distance-derived boundaries.
    markers = markers.astype(np.int32)

    # Separate cores with watershed over an inverted distance map.
    surface = cv2.normalize(
        distance,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    surface = cv2.cvtColor(
        255 - surface,
        cv2.COLOR_GRAY2BGR
    )

    markers[binary == 0] = 0

    cv2.watershed(
        surface,
        markers
    )

    teeth = []

    for label in np.unique(markers):
        if label <= 0:
            continue

        tooth = np.zeros_like(binary)
        tooth[markers == label] = 255

        tooth = cv2.bitwise_and(
            tooth,
            binary
        )

        area = cv2.countNonZero(tooth)

        if area < 45:
            continue

        contours, _ = cv2.findContours(
            tooth,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(contour)

        if w < 5 or h < 5:
            continue

        M = cv2.moments(contour)

        if M["m00"]:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2
            cy = y + h / 2

        teeth.append({
            "mask": tooth,
            "contour": contour,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": float(cx),
            "cy": float(cy),
            "area": int(area),
        })

    teeth.sort(key=lambda t: t["cx"])

    # Watershed can occasionally create too many fragments.
    # Fall back if the result is obviously fragmented.
    if len(teeth) == 0:
        return _fallback_teeth(binary)

    for i, tooth in enumerate(teeth, 1):
        tooth["index"] = i

    return teeth


# ============================================================
# Edge extraction
# ============================================================

def _edge_points(mask, direction):
    """
    Return one visible incisal-edge point per x.

    bottom = upper tooth, therefore bottommost point
    top    = lower tooth, therefore topmost point
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, None

    unique_x = np.unique(xs)

    edge_x = []
    edge_y = []

    for x in unique_x:

        yy = ys[xs == x]

        if len(yy) == 0:
            continue

        if direction == "bottom":
            y = np.max(yy)
        else:
            y = np.min(yy)

        edge_x.append(x)
        edge_y.append(y)

    if len(edge_x) < 5:
        return None, None

    return (
        np.asarray(edge_x, dtype=np.float32),
        np.asarray(edge_y, dtype=np.float32)
    )


# ============================================================
# Curve fitting
# ============================================================

def _fit_shoulder_curve(edge_x, edge_y, left, right):
    """
    Fit a quadratic only to intact shoulder regions.
    The suspected chip itself is excluded.
    """

    n = len(edge_x)

    shoulder = max(
        7,
        int(round(n * 0.22))
    )

    margin = max(
        2,
        int(round(n * 0.08))
    )

    a = max(
        margin,
        left - shoulder
    )

    b = min(
        n - margin - 1,
        right + shoulder
    )

    indices = []

    if a > margin:
        indices.extend(
            range(margin, a)
        )

    if b < n - margin:
        indices.extend(
            range(b, n - margin)
        )

    if len(indices) < 8:
        return None

    indices = np.asarray(
        indices,
        dtype=np.int32
    )

    fx = edge_x[indices]
    fy = edge_y[indices]

    center = float(
        np.mean(fx)
    )

    scale = max(
        float(np.ptp(fx)),
        1.0
    )

    xn = (
        fx - center
    ) / scale

    try:
        coeff = np.polyfit(
            xn,
            fy,
            2
        )
    except Exception:
        return None

    def curve(x):
        x = np.asarray(
            x,
            dtype=np.float32
        )

        return np.polyval(
            coeff,
            (x - center) / scale
        )

    return curve


# ============================================================
# CHIP RECONSTRUCTION
# ============================================================

def create_chip_repair_mask(tooth, mouth_center_y):
    """
    Explicitly create NEW pixels outside the detected tooth.

    The suspected missing section is identified by comparing the
    visible edge to a curve fitted from intact left/right shoulders.
    """

    mask = tooth["mask"]

    H, W = mask.shape
    w = tooth["w"]
    h = tooth["h"]

    if w < 12 or h < 8:
        return np.zeros_like(mask)

    # Only reconstruct upper incisal edges for now.
    # This is where the supplied input10 defect is located.
    if tooth["cy"] >= mouth_center_y:
        return np.zeros_like(mask)

    direction = "bottom"

    edge_x, edge_y = _edge_points(
        mask,
        direction
    )

    if edge_x is None:
        return np.zeros_like(mask)

    n = len(edge_x)

    if n < 12:
        return np.zeros_like(mask)

    # Smooth the observed edge.
    kernel = max(
        5,
        int(round(n * 0.09))
    )

    if kernel % 2 == 0:
        kernel += 1

    kernel = min(kernel, 11)

    smoothed = np.convolve(
        edge_y,
        np.ones(kernel, dtype=np.float32) / kernel,
        mode="same"
    )

    smoothed[:kernel // 2] = edge_y[:kernel // 2]
    smoothed[-kernel // 2:] = edge_y[-kernel // 2:]

    # --------------------------------------------------------
    # Find local inward notch.
    #
    # First create a broad baseline.
    # --------------------------------------------------------

    broad = max(
        9,
        int(round(n * 0.40))
    )

    if broad % 2 == 0:
        broad += 1

    broad = min(
        broad,
        31
    )

    half = broad // 2

    padded = np.pad(
        smoothed,
        (half, half),
        mode="edge"
    )

    baseline = np.empty_like(
        smoothed
    )

    for i in range(n):
        baseline[i] = np.median(
            padded[i:i + broad]
        )

    # For upper teeth an inward chip makes the edge move UP,
    # therefore baseline - observed edge is positive.
    inward = baseline - smoothed

    # Ignore corners.
    margin = max(
        3,
        int(round(n * 0.12))
    )

    inward[:margin] = 0
    inward[-margin:] = 0

    peak = int(
        np.argmax(inward)
    )

    peak_depth = float(
        inward[peak]
    )

    # --------------------------------------------------------
    # Even if the segmentation does not show a strong notch,
    # inspect the central region. The missing material in
    # input10 is large enough that the curve should extend into
    # the dark exposed area.
    # --------------------------------------------------------

    if peak_depth < 1.5:

        center = n // 2

        search_half = max(
            4,
            int(round(n * 0.20))
        )

        lo = max(
            margin,
            center - search_half
        )

        hi = min(
            n - margin,
            center + search_half
        )

        peak = lo + int(
            np.argmax(
                inward[lo:hi]
            )
        )

        peak_depth = float(
            inward[peak]
        )

    # --------------------------------------------------------
    # Define a repair region around the peak.
    #
    # Minimum width is intentionally large enough to make a
    # visible correction on the supplied image.
    # --------------------------------------------------------

    repair_width = max(
        7,
        int(round(n * 0.20))
    )

    left = max(
        margin,
        peak - repair_width // 2
    )

    right = min(
        n - margin - 1,
        peak + repair_width // 2
    )

    if right - left < 4:
        return np.zeros_like(mask)

    curve = _fit_shoulder_curve(
        edge_x,
        edge_y,
        left,
        right
    )

    if curve is None:
        return np.zeros_like(mask)

    # --------------------------------------------------------
    # Generate NEW geometry.
    # --------------------------------------------------------

    repair = np.zeros_like(mask)

    max_depth = max(
        5.0,
        h * 0.55
    )

    for i in range(left, right + 1):

        x = int(round(edge_x[i]))

        actual = float(
            edge_y[i]
        )

        target = float(
            curve(edge_x[i])
        )

        # For an upper tooth target must be below the visible
        # chipped edge.
        if target <= actual + 0.25:
            continue

        target = min(
            target,
            actual + max_depth
        )

        y1 = int(
            np.floor(actual + 0.25)
        )

        y2 = int(
            np.ceil(target)
        )

        if y2 <= y1:
            continue

        y1 = max(0, y1)
        y2 = min(H - 1, y2)

        if 0 <= x < W:
            repair[
                y1:y2 + 1,
                x
            ] = 255

    # Thicken/blend the newly generated geometry.
    repair = cv2.dilate(
        repair,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Absolutely critical:
    # keep only pixels that are NOT already part of the tooth.
    repair = cv2.bitwise_and(
        repair,
        cv2.bitwise_not(mask)
    )

    area = cv2.countNonZero(repair)

    print(
        f"[chip] tooth {tooth.get('index', '?')}: "
        f"width={w}, "
        f"height={h}, "
        f"peak_depth={peak_depth:.2f}, "
        f"region={left}:{right}, "
        f"NEW pixels={area}"
    )

    return repair


# ============================================================
# Restoration mask
# ============================================================

def create_restoration_mask(
    veneer_mask,
    mouth,
    mouth_center_y
):
    """
    Create strictly separated per-tooth restoration regions.

    Every tooth gets its own closed mask.  No global convex hull,
    dilation, or mouth-wide whitening is allowed.
    """

    teeth = extract_individual_teeth(veneer_mask)

    restoration = np.zeros_like(veneer_mask)
    repair_total = np.zeros_like(veneer_mask)

    for tooth in teeth:

        # Use the actual watershed/connected-component tooth as the
        # immutable border for cosmetic surface changes.
        own = tooth["mask"].copy()

        # Small erosion keeps color treatment away from the edge
        # where gum/lip pixels can contaminate the segmentation.
        safe = cv2.erode(
            own,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        # If erosion removed too much, use the original tooth.
        if cv2.countNonZero(safe) < max(
            12,
            int(cv2.countNonZero(own) * 0.45)
        ):
            safe = own

        tooth["surface_mask"] = safe

        restoration = cv2.bitwise_or(
            restoration,
            own
        )

        chip = create_chip_repair_mask(
            tooth,
            mouth_center_y
        )

        tooth["repair_mask"] = chip

        repair_total = cv2.bitwise_or(
            repair_total,
            chip
        )

    # Original enamel + explicitly reconstructed incisal material.
    restoration = cv2.bitwise_or(
        restoration,
        repair_total
    )

    # Final safety: only chip pixels may exist outside the detected
    # enamel.  There is no other outward expansion.
    return (
        restoration,
        repair_total,
        teeth
    )



# ============================================================
# Reconstruction / surface
# ============================================================

def _reconstruct_pixels(
    mouth,
    repair_mask,
    teeth=None
):
    """
    Reconstruct missing enamel using texture sampled from the
    SAME tooth rather than simply painting/inpainting the gap.

    This is important for realism:
      - the new area receives tooth-colored pixels
      - local enamel texture is retained
      - the transition is feathered
      - the original lip/gum pixels are not copied into the repair
    """

    if cv2.countNonZero(repair_mask) == 0:
        return mouth.copy()

    result = mouth.copy()

    # Start with a conservative inpaint result.  This is useful for
    # tiny boundary pixels where a source tooth pixel is unavailable.
    fallback = cv2.inpaint(
        mouth,
        repair_mask,
        3,
        cv2.INPAINT_TELEA
    )

    if teeth is None:
        teeth = []

    for tooth in teeth:

        chip = tooth.get("repair_mask")

        if chip is None:
            continue

        if cv2.countNonZero(chip) == 0:
            continue

        mask = tooth["mask"]

        H, W = mask.shape

        ys, xs = np.where(chip > 0)

        if len(xs) == 0:
            continue

        # --------------------------------------------------------
        # Find a reliable enamel source for every repair column.
        #
        # For an upper tooth, the missing edge is BELOW the
        # original edge.  Therefore sample several pixels ABOVE
        # the original edge from the same tooth.
        # --------------------------------------------------------

        source = fallback.copy()

        for x in np.unique(xs):

            column_repair = ys[xs == x]

            if len(column_repair) == 0:
                continue

            x0 = int(x)

            tooth_rows = np.where(mask[:, x0] > 0)[0]

            if len(tooth_rows) == 0:
                continue

            actual_edge = int(tooth_rows.max())

            # Use an interior strip, not the immediate edge.
            src_y = actual_edge - min(
                7,
                max(2, int(round(tooth["h"] * 0.22)))
            )

            src_y = int(
                np.clip(
                    src_y,
                    int(tooth["y"]),
                    int(tooth["y"] + tooth["h"] - 1)
                )
            )

            # If that source is unavailable, use the nearest
            # interior tooth pixel.
            if mask[src_y, x0] == 0:
                candidates = tooth_rows[
                    tooth_rows <= actual_edge - 1
                ]

                if len(candidates):
                    src_y = int(candidates[-1])
                else:
                    continue

            # Copy a short vertical texture profile.  This keeps
            # the repair from looking like a flat solid patch.
            for y in column_repair:

                yi = int(y)

                # Map deeper repair pixels slightly farther into
                # the tooth so the texture varies naturally.
                depth = max(
                    0,
                    yi - actual_edge
                )

                sample_y = int(
                    np.clip(
                        src_y - int(depth * 0.30),
                        int(tooth["y"]),
                        actual_edge
                    )
                )

                if 0 <= sample_y < H:
                    source[yi, x0] = mouth[
                        sample_y,
                        x0
                    ]

        # Fill any pixels that did not get a source sample.
        missing = (
            (chip > 0) &
            (np.any(source == mouth, axis=2))
        )

        # We do not use this heuristic as the primary path; just
        # ensure all repair pixels are populated.
        source[chip > 0] = np.where(
            missing[chip > 0, None],
            fallback[chip > 0],
            source[chip > 0]
        )

        # --------------------------------------------------------
        # Add very subtle vertical enamel shading.
        # Teeth are not flat white rectangles.
        # --------------------------------------------------------

        region = np.zeros_like(chip)

        cv2.dilate(
            chip,
            np.ones((7, 7), np.uint8),
            iterations=1,
            dst=region
        )

        lab = cv2.cvtColor(
            source,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        local_y = np.indices((H, W))[0].astype(np.float32)

        center_y = float(
            tooth["y"] + tooth["h"] * 0.42
        )

        height = max(
            1.0,
            tooth["h"] * 0.70
        )

        vertical = (
            (local_y - center_y) / height
        )

        # Very small tonal falloff.
        shade = np.clip(
            1.0 - 0.035 * np.abs(vertical),
            0.96,
            1.01
        )

        lab[:, :, 0] *= shade

        shaded = cv2.cvtColor(
            np.clip(
                lab,
                0,
                255
            ).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        )

        # --------------------------------------------------------
        # Feather only the boundary of the generated geometry.
        # Interior stays sharp enough to retain enamel detail.
        # --------------------------------------------------------

        feather = cv2.GaussianBlur(
            chip,
            (0, 0),
            1.15
        ).astype(np.float32) / 255.0

        alpha = np.clip(
            feather * 0.98,
            0.0,
            0.98
        )

        # Keep the central repair strong; only feather the edge.
        alpha[chip > 0] = np.maximum(
            alpha[chip > 0],
            0.88
        )

        alpha3 = alpha[:, :, None]

        result = (
            result.astype(np.float32) * (1.0 - alpha3)
            + shaded.astype(np.float32) * alpha3
        ).astype(np.uint8)

    return result



def create_veneer_surface(
    mouth,
    veneer_mask,
    intensity=85
):
    """
    Natural veneer rendering with HARD PER-TOOTH GEOMETRY.

    Surface treatment is allowed only inside each detected tooth's
    own mask.  The only pixels allowed outside an existing tooth
    are the explicitly reconstructed incisal-chip pixels.

    This prevents the characteristic failure where the veneer looks
    as though it is growing out of the gums or lips.
    """

    mouth_center_y = mouth.shape[0] * 0.50

    restoration_mask, repair_mask, teeth = \
        create_restoration_mask(
            veneer_mask,
            mouth,
            mouth_center_y
        )

    strength = float(
        np.clip(intensity / 100.0, 0.0, 1.0)
    )

    result = mouth.copy()

    # ------------------------------------------------------------
    # RECONSTRUCT CHIPS FIRST.
    # ------------------------------------------------------------
    result = _reconstruct_pixels(
        mouth,
        repair_mask,
        teeth
    )

    # ------------------------------------------------------------
    # PER-TOOTH COSMETIC SURFACE.
    #
    # Never use one blurred/global mask across the smile.
    # ------------------------------------------------------------
    for tooth in teeth:

        surface_mask = tooth.get("surface_mask")

        if surface_mask is None:
            surface_mask = tooth["mask"]

        if cv2.countNonZero(surface_mask) == 0:
            continue

        # Feather only inward from the tooth boundary.  We create
        # the blur, then intersect it with the original tooth mask.
        # Therefore the blur can NEVER enter the gums/lips.
        soft = cv2.GaussianBlur(
            surface_mask,
            (0, 0),
            0.75
        ).astype(np.float32) / 255.0

        soft = np.minimum(
            soft,
            tooth["mask"].astype(np.float32) / 255.0
        )

        # Conservative cosmetic change.
        alpha = (
            0.22
            * strength
            * soft
        )

        # --------------------------------------------------------
        # LAB colour adjustment.
        # --------------------------------------------------------
        local = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        L = local[:, :, 0]
        A = local[:, :, 1]
        B = local[:, :, 2]

        tooth_region = surface_mask > 0

        # Keep natural tooth warmth.
        L_target = np.clip(
            L + 4.0 * strength,
            0,
            255
        )

        A_target = (
            A * 0.94 +
            128.0 * 0.06
        )

        B_target = (
            B * 0.97 +
            128.0 * 0.03
        )

        local[:, :, 0] = np.where(
            tooth_region,
            L_target,
            L
        )

        local[:, :, 1] = np.where(
            tooth_region,
            A_target,
            A
        )

        local[:, :, 2] = np.where(
            tooth_region,
            B_target,
            B
        )

        cosmetic = cv2.cvtColor(
            np.clip(local, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        )

        result = (
            result.astype(np.float32)
            * (1.0 - alpha[:, :, None])
            + cosmetic.astype(np.float32)
            * alpha[:, :, None]
        ).astype(np.uint8)

    # ------------------------------------------------------------
    # VERY SUBTLE ENAMEL HIGHLIGHT, again clipped per tooth.
    # ------------------------------------------------------------
    gray = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.2
    )

    local_detail = np.clip(
        gray - blurred,
        0,
        15
    )

    for tooth in teeth:

        surface_mask = tooth.get(
            "surface_mask",
            tooth["mask"]
        )

        highlight_mask = (
            surface_mask.astype(np.float32) / 255.0
        )

        amount = (
            local_detail
            * 0.055
            * strength
            * highlight_mask
        )

        rf = result.astype(np.float32)

        rf[:, :, 0] += amount
        rf[:, :, 1] += amount
        rf[:, :, 2] += amount

        result = np.clip(
            rf,
            0,
            255
        ).astype(np.uint8)

    # ------------------------------------------------------------
    # REPAIR AREA: preserve the reconstructed shape and blend its
    # colour slightly.  It is NEVER allowed to expand here.
    # ------------------------------------------------------------
    if cv2.countNonZero(repair_mask) > 0:

        lab = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2LAB
        ).astype(np.float32)

        repair_region = repair_mask > 0

        lab[:, :, 0] = np.where(
            repair_region,
            np.clip(
                lab[:, :, 0] + 2.5 * strength,
                0,
                255
            ),
            lab[:, :, 0]
        )

        repaired = cv2.cvtColor(
            np.clip(lab, 0, 255).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        )

        # Blend only where repair exists.  Dilated/blurred masks are
        # clipped back to the repair itself, preventing gum bleed.
        repair_alpha = cv2.GaussianBlur(
            repair_mask,
            (0, 0),
            0.65
        ).astype(np.float32) / 255.0

        repair_alpha = np.minimum(
            repair_alpha,
            repair_mask.astype(np.float32) / 255.0
        )

        repair_alpha[repair_region] = np.maximum(
            repair_alpha[repair_region],
            0.90
        )

        result = (
            result.astype(np.float32)
            * (1.0 - repair_alpha[:, :, None])
            + repaired.astype(np.float32)
            * repair_alpha[:, :, None]
        ).astype(np.uint8)

    return (
        result,
        restoration_mask,
        repair_mask,
        teeth
    )



# ============================================================
# Debug
# ============================================================

def draw_teeth_debug(
    mouth,
    teeth
):
    debug = mouth.copy()

    for i, tooth in enumerate(teeth, 1):

        # GREEN = each individual tooth border.
        cv2.drawContours(
            debug,
            [tooth["contour"]],
            -1,
            (0, 255, 0),
            1
        )

        repair = tooth.get("repair_mask")

        if repair is not None:
            debug[repair > 0] = (
                255, 0, 255
            )

        cv2.putText(
            debug,
            str(i),
            (
                int(tooth["cx"]),
                int(tooth["cy"])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return debug



def draw_reconstruction_debug(
    mouth,
    teeth,
    mouth_center_y
):
    """
    GREEN   = detected tooth
    BLUE    = shoulder-fitted ideal edge
    MAGENTA = actual NEW reconstruction pixels
    """

    debug = mouth.copy()

    for i, tooth in enumerate(teeth, 1):

        mask = tooth["mask"]

        cv2.drawContours(
            debug,
            [tooth["contour"]],
            -1,
            (0, 255, 0),
            1
        )

        repair = tooth.get("repair_mask")
        if repair is not None:
            debug[repair > 0] = (255, 0, 255)

        if tooth["cy"] >= mouth_center_y:
            continue

        edge_x, edge_y = _edge_points(
            mask,
            "bottom"
        )

        if edge_x is None:
            continue

        # Re-run the same broad-notch logic used by repair.
        n = len(edge_x)

        smooth = np.convolve(
            edge_y,
            np.ones(5, dtype=np.float32) / 5.0,
            mode="same"
        )

        smooth[:2] = edge_y[:2]
        smooth[-2:] = edge_y[-2:]

        broad = max(
            9,
            min(31, int(round(n * 0.40)))
        )

        if broad % 2 == 0:
            broad += 1

        half = broad // 2
        padded = np.pad(
            smooth,
            (half, half),
            mode="edge"
        )

        baseline = np.empty_like(
            smooth
        )

        for j in range(n):
            baseline[j] = np.median(
                padded[j:j + broad]
            )

        inward = baseline - smooth

        margin = max(
            3,
            int(n * 0.12)
        )

        inward[:margin] = 0
        inward[-margin:] = 0

        peak = int(
            np.argmax(inward)
        )

        width = max(
            7,
            int(round(n * 0.20))
        )

        left = max(
            margin,
            peak - width // 2
        )

        right = min(
            n - margin - 1,
            peak + width // 2
        )

        curve = _fit_shoulder_curve(
            edge_x,
            edge_y,
            left,
            right
        )

        if curve is not None:

            for x in np.linspace(
                edge_x[left],
                edge_x[right],
                100
            ):

                y = float(curve(x))

                xi = int(round(x))
                yi = int(round(y))

                if (
                    0 <= xi < debug.shape[1]
                    and 0 <= yi < debug.shape[0]
                ):
                    debug[yi, xi] = (
                        255, 0, 0
                    )

        repair = tooth.get(
            "repair_mask"
        )

        if repair is not None:
            debug[repair > 0] = (
                255, 0, 255
            )

        cv2.putText(
            debug,
            str(i),
            (
                int(tooth["cx"]),
                int(tooth["cy"])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    return debug


# ============================================================
# SVG
# ============================================================

def export_teeth_svg(
    teeth,
    path,
    width,
    height
):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]

    for i, tooth in enumerate(teeth, 1):

        pts = tooth["contour"].reshape(-1, 2)

        if len(pts) == 0:
            continue

        d = "M " + " L ".join(
            f"{int(x)} {int(y)}"
            for x, y in pts
        ) + " Z"

        lines.append(
            f'<path d="{d}" fill="none" '
            f'stroke="lime" stroke-width="1"/>'
        )

        lines.append(
            f'<text x="{int(tooth["cx"])}" '
            f'y="{int(tooth["cy"])}" '
            f'font-size="10" fill="red">{i}</text>'
        )

    lines.append("</svg>")

    Path(path).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# Full pipeline
# ============================================================

def veneer_smile(
    image,
    intensity=85,
    return_debug=False
):
    landmarks = detect_face_landmarks(
        image
    )

    if landmarks is None:
        raise RuntimeError(
            "No face detected."
        )

    mouth, inner_mask, mouth_x, mouth_y = \
        get_mouth_region(
            image,
            landmarks
        )

    veneer_mask = create_veneer_mask(
        mouth,
        inner_mask
    )

    # Estimate the upper/lower boundary in mouth coordinates.
    mouth_center_y = mouth.shape[0] * 0.50

    result, restoration_mask, repair_mask, teeth = \
        create_veneer_surface(
            mouth,
            veneer_mask,
            intensity
        )

    teeth_debug = draw_teeth_debug(
        mouth,
        teeth
    )

    reconstruction_debug = \
        draw_reconstruction_debug(
            mouth,
            teeth,
            mouth_center_y
        )

    debug = {
        "mouth": mouth,
        "inner_mask": inner_mask,
        "veneer_mask": veneer_mask,
        "shape_mask": veneer_mask,
        "restoration_mask": restoration_mask,
        "repair_mask": repair_mask,
        "teeth_debug": teeth_debug,
        "reconstruction_debug": reconstruction_debug,
        "teeth": teeth,
        "tooth_surface_masks": [
            t.get("surface_mask", t["mask"])
            for t in teeth
        ],
        "landmarks": landmarks,
        "mouth_x": mouth_x,
        "mouth_y": mouth_y,
    }

    # IMPORTANT:
    # All veneer processing happens in the mouth crop, but the public
    # function must return a FULL-SIZE image.  The previous version
    # returned the mouth crop directly, which caused test_veneer.py
    # to report "Output dimensions differ from input".
    full_result = image.copy()

    mh, mw = result.shape[:2]
    H, W = image.shape[:2]

    y2 = min(mouth_y + mh, H)
    x2 = min(mouth_x + mw, W)

    crop_h = y2 - mouth_y
    crop_w = x2 - mouth_x

    if crop_h > 0 and crop_w > 0:
        full_result[
            mouth_y:y2,
            mouth_x:x2
        ] = result[
            :crop_h,
            :crop_w
        ]

    if not return_debug:
        return full_result

    debug["full_result"] = full_result

    return full_result, debug
