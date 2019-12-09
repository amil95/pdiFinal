import cv2
import numpy as np
import dft as dft
import connectedcomponents as cc
from matplotlib import pyplot as plt
import os
import operator

#find characters using contour
def findCharactersUsingContours(binary_image): # Encontrar cada caracteres utilizando identificacao de blobs por FindContours
    #cv2.imshow("a", binary_image)
    cnts, hier = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cnts = list of contours
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] # bounding Box of each character
    characters = cnts
    #print(characters)
    boundingBoxes = sorted(boundingBoxes, key=operator.itemgetter(1, 0)) # ordena os chars da esquerda para a direita e de cima para baixo
    if __debug__:
        for boundingBox in boundingBoxes:
            x,y,w,h = boundingBox
            roi = im_th[y:y+h, x:x+w] # the same, for a single character
            cv2.rectangle(binary_image,(x,y),(x+w,y+h),(255,255,255),2)
            #cv2.imshow("a", binary_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    return characters, boundingBoxes

def templateMatching(image, template):
    result = cv2.matchTemplate(np.bitwise_not(image), template, cv2.TM_SQDIFF) # template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get values
    #if __debug__:
        #print(result)
        #result = result*255
        #cv2.imshow("Template matching result", result)
    return min_val, max_val, min_loc, max_loc # return values

def buildCharacterMap():
    alphabet = cv2.imread("alphabetALLCAPS.png")
    gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
    th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary
    characters, boundingBoxes = findCharactersUsingContours(np.bitwise_not(alphabet_th))
    charactermap = np.zeros((26,2,2))
    i=0;
    for box in boundingBoxes:
        x,y,w,h = box #the same, for a single character
        roi = alphabet_th[y:y+h, x:x+w] # the same, for a single character
        min_val, max_val, min_loc, max_loc = templateMatching(alphabet_th, roi)
        charactermap[i][0] = max_loc
        charactermap[i][1] = min_loc
        i+=1
    return charactermap



img = cv2.imread("testessss.png")
alphabet = cv2.imread("alfabeto2.png")
threshold = 200

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grayscale
gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binary image
th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary

charactermap = buildCharacterMap()
#print(charactermap)
characters, boundingBoxes = findCharactersUsingContours(np.bitwise_not(im_th))

text = ""
ranger = 5

img2 = im_th.copy()
temp = 0

for box in boundingBoxes:
    x,y,w,h = box #the same, for a single character
    roi = im_th[y:y+h, x:x+w] # the same, for a single character
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get location for highest score
    w, h = roi.shape[::-1]  #get shape of roi
    top_left = min_loc  #top left of rectangle
    bottom_right = (top_left[0] + w, top_left[1] + h) #bottom_right of rectangle
    alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters
    cv2.rectangle(alphabet_highlight, top_left, bottom_right, (0,0,0), 2)

    if (temp-x) >= 29:
        text = text + " "
    #print(temp - x, temp, x)
    temp = x
    i=0
    for i in range(len(charactermap)):
        print(charactermap[i][0][0], max_loc[0], charactermap[i][0][1], max_loc[1], charactermap[i][0][0], max_loc[0]-ranger, charactermap[i][0][1], max_loc[1]-ranger, sep=", ")
        if charactermap[i][0][0] <= max_loc[0]+ranger and charactermap[i][0][1] <= max_loc[1]+ranger and charactermap[i][0][0] >= max_loc[0]-ranger and charactermap[i][0][1] >= max_loc[1]-ranger:
            text = text + str(chr(i+65))
        i+=1
print(text[::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

#
#     meth = "Heisenberg"

    # plt.subplot(121),plt.imshow(result,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(gray_alphabet,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()
