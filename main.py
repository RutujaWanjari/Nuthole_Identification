import cv2
import numpy as np


# Read input image
image = cv2.imread('input.jpg')

# Convert image into gray scaled image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert gray scaled image into binary image
binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)[1]

# Set size of kernel.
kernel = np.ones((7, 7), np.uint8)

# It's a erosion followed by dilation, basically removes noise.
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Find contours
_, contours0, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Approximate the contours
contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

for cnt in contours:
    area = cv2.contourArea(cnt)
    # Differentiate appropriate contours(nut holes) from normal holes
    if area<250 and area>50:
        # Highlight contours
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        center = (int(x), int(y))
        cv2.circle(image, center, 10, (0, 255, 0), 2)

# Display output
cv2.imshow('output', image)
cv2.waitKey()