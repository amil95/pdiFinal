import cv2
import numpy as np

img = cv2.imread("testImage.png", 0)
threshold = 200

# img[img < threshold] = 0
# img[img >= threshold] = 255

th, im_th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

im_floodfill = im_th.copy()

h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

cv2.floodFill(im_floodfill, mask, (0, 0), 255)

im_floodfill_inv = cv2.bitwise_not(im_floodfill)

fill_image = im_th | im_floodfill_inv


cv2.imshow("", fill_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
