import cv2
import numpy as np

# Read the input image
image_path = 'boxes.jpg'  # Replace with the actual image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Define the lower and upper bounds for detecting brown color
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([30, 255, 255])

# Convert to HSV color space and create a mask for brown color
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_brown, upper_brown)

# Apply Canny edge detection
edges = cv2.Canny(blurred, threshold1=30, threshold2=150)

# Combine the edge mask and the brown mask
combined_mask = cv2.bitwise_and(edges, mask)

# Find contours in the combined mask
contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding rectangles around the detected boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w > 50 and h > 50:  # Adjust these values to fit your boxes
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Detected Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()