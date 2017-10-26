import cv2
import numpy as np


# Read input image
image = cv2.imread('input.jpg')

# Convert image into gray scaled image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert gray scaled image into binary image
binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

_, contours0, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area<400:
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2)

cv2.imshow('output', image)
cv2.waitKey()