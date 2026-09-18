import cv2
from pathlib import Path

from alignment import (
    detect_face_landmarks,
    get_mouth_region,
    create_tooth_mask,
    straighten_teeth
)


BASE_DIR = Path(__file__).resolve().parents[1]

input_path = BASE_DIR / "samples" / "input9.jpg"
output_path = BASE_DIR / "samples" / "aligned_output.jpg"
mask_path = BASE_DIR / "samples" / "alignment_mask.jpg"
mouth_path = BASE_DIR / "samples" / "alignment_mouth.jpg"


image = cv2.imread(str(input_path))

if image is None:
    raise ValueError(f"Could not read image: {input_path}")


landmarks = detect_face_landmarks(image)

if landmarks is None:
    raise ValueError("No face detected!")


mouth, inner_mask, x_min, y_min, x_max, y_max = get_mouth_region(
    image,
    landmarks
)


mask = create_tooth_mask(mouth, inner_mask)


print("Mouth size:", mouth.shape)
print("Tooth pixels detected:", cv2.countNonZero(mask))


cv2.imwrite(
    str(mouth_path),
    mouth
)

cv2.imwrite(
    str(mask_path),
    mask
)


result_mouth = straighten_teeth(
    mouth,
    mask,
    100
)


output = image.copy()

output[
    y_min:y_max,
    x_min:x_max
] = result_mouth


cv2.imwrite(
    str(output_path),
    output
)


print("Alignment test completed!")
print(f"Mask saved to: {mask_path}")
print(f"Output saved to: {output_path}")