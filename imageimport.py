import cv2
import numpy as np


def importImages(image, alphabet):
    #img = cv2.imread("mono.jpg")
    img = cv2.imread(image)
    #alphabet = cv2.imread("testimage2.png")
    threshold = 200
    alphabet = cv2.imread(alphabet)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grayscale
    gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
    th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binary image
    th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary

    return img, im_th, alphabet_th
