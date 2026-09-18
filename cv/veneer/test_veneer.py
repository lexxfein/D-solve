from pathlib import Path
import cv2
import numpy as np

from veneer import veneer_smile, export_teeth_svg


HERE = Path(__file__).resolve()
CV_DIR = HERE.parents[1]

INPUT = CV_DIR / "samples" / "input10.jpg"
OUT = CV_DIR / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, image):
    path = OUT / name

    if image is None:
        raise RuntimeError(f"Image is None: {name}")

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save: {path}")

    print(f"Saved: {path}")


image = cv2.imread(str(INPUT))

if image is None:
    raise FileNotFoundError(
        f"Could not read input:\n{INPUT}"
    )


result, debug = veneer_smile(
    image,
    intensity=85,
    return_debug=True
)


save("veneer_output.jpg", result)
save("veneer_mouth.jpg", debug["mouth"])
save("veneer_inner_mask.jpg", debug["inner_mask"])
save("veneer_mask.jpg", debug["veneer_mask"])
save("veneer_shape_mask.jpg", debug["shape_mask"])
save("veneer_restoration_mask.jpg", debug["restoration_mask"])
save("veneer_repair_mask.jpg", debug["repair_mask"])
save("veneer_teeth_debug.jpg", debug["teeth_debug"])
save("veneer_reconstruction_debug.jpg", debug["reconstruction_debug"])


# One combined visualization of the individual per-tooth masks.
combined = np.zeros_like(debug["mouth"])
for idx, mask in enumerate(debug["tooth_surface_masks"], start=1):
    combined[mask > 0] = debug["mouth"][mask > 0]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    cv2.drawContours(
        combined,
        contours,
        -1,
        (0, 255, 0),
        1
    )

    ys, xs = np.where(mask > 0)
    if len(xs):
        cv2.putText(
            combined,
            str(idx),
            (int(xs.mean()), int(ys.mean())),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

save("veneer_individual_tooth_masks.jpg", combined)


export_teeth_svg(
    debug["teeth"],
    OUT / "veneer_teeth.svg",
    debug["mouth"].shape[1],
    debug["mouth"].shape[0]
)

diff = cv2.absdiff(image, result)

changed_pixels = int(
    np.count_nonzero(
        np.any(diff > 0, axis=2)
    )
)

mean_difference = float(diff.mean())

repair_pixels = int(
    cv2.countNonZero(
        debug["repair_mask"]
    )
)

print()
print("=" * 65)
print("STRICT INDIVIDUAL-TOOTH VENEER SIMULATION")
print("=" * 65)
print(f"Input shape:     {image.shape}")
print(f"Output shape:    {result.shape}")
print(f"Detected teeth:  {len(debug['teeth'])}")
print(f"Repair pixels:   {repair_pixels}")
print(f"Changed pixels:  {changed_pixels}")
print(f"Mean difference: {mean_difference:.6f}")
print("=" * 65)

if result.shape != image.shape:
    raise RuntimeError(
        "Output dimensions differ from input."
    )

if changed_pixels == 0:
    raise RuntimeError(
        "Output is pixel-identical to input10.jpg."
    )

print()
print("OUTPUT:")
print(OUT / "veneer_output.jpg")
print()
print("INDIVIDUAL TOOTH DEBUG:")
print(OUT / "veneer_individual_tooth_masks.jpg")
print()
print("RECONSTRUCTION DEBUG:")
print(OUT / "veneer_reconstruction_debug.jpg")
