import cv2
from whitening import whiten_smile


# -----------------------------------------
# Change this image whenever you want
# -----------------------------------------

image_path = "cv/samples/input1.jpg"


# -----------------------------------------
# Whitening strength
# -----------------------------------------

intensity = 100


# -----------------------------------------
# Load image
# -----------------------------------------

image = cv2.imread(
    image_path
)

if image is None:
    print("Could not load image")
    exit()

print("Input loaded!")
print("Input size:", image.shape)


# -----------------------------------------
# Whitening
# -----------------------------------------

try:

    result = whiten_smile(
        image,
        intensity
    )

except Exception as e:

    print("Whitening failed:")
    print(e)
    exit()


# -----------------------------------------
# Save
# -----------------------------------------

cv2.imwrite(
    "cv/whitening/final_whitened.jpg",
    result
)

print("Whitening successful!")
print("Intensity:", intensity)
print("Output size:", result.shape)
print(
    "Saved: cv/whitening/final_whitened.jpg"
)