#python modules
import cv2
import numpy as np
import dft as dft
import connectedcomponents as cc
from matplotlib import pyplot as plt
import os
#custom modules
import math
import hough
import imageimport as imgimp
import operations as op

def waitKey(b):
    a = cv2.waitKey(b)
    if a == 49:
        exit(0)
    if a == 50:
        return 1
    else:
        return 0

string1 = "texto_roboto_mono_24.png"
string2 = "alfabeto_roboto_mono.png"

original, im_th, alphabet_th = imgimp.importImages(string1, string2)

# cv2.imshow("Original", original)
# waitKey()
# cv2.destroyAllWindows()
# cv2.imshow("Binary", im_th)
# waitKey()
# cv2.destroyAllWindows()

im_th, im_lines, text_lines, im_aggregated = hough.unrotateImage(im_th)

cv2.imshow("Hough Lines", im_lines)
waitKey()
cv2.destroyAllWindows()

cv2.imshow("Hough Lines aggregated", im_aggregated)
waitKey()
cv2.destroyAllWindows()

# cv2.imshow("Correct rotation", im_th)
# waitKey()
# cv2.destroyAllWindows()

characters, boundingBoxes, min_x, min_y = op.findCharactersUsingContours(np.bitwise_not(im_th))

text = ""
ranger = 5

img2 = im_th.copy()
temp = 0

gridSlotSizeX = 18
gridSlotSizeY = 31
#charactermap = op.buildCharacterMap(im_th)
a = 0

for box in boundingBoxes:
    x,y,w,h = box
    if w < 9 or h < 10:
        continue
    startPointX = min_x - int((gridSlotSizeX - w)/2 + 0.5)
    startPointY = min_y - int((gridSlotSizeY - h)/2 + 0.5) + 4
    cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
    waitKey(0)
    roi = im_th[y:y+h, x:x+w]
    while(len(img2) > startPointY + gridSlotSizeY):
        startPointX = x - int((gridSlotSizeX - w)/2 + 0.5)
        while(len(img2[0]) > startPointX + gridSlotSizeX):
            alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters
            roi = im_th[startPointY:startPointY + gridSlotSizeY, startPointX:startPointX + gridSlotSizeX]
            cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
            cv2.imshow("Text", img2)
            result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get location for highest score
            w, h = roi.shape[::-1]  #get shape of roi
            top_left = max_loc  #top left of rectangle
            bottom_right = (top_left[0] + w, top_left[1] + h) #bottom_right of rectangle
            alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters
            cv2.rectangle(alphabet_highlight, top_left, bottom_right, (0,0,0), 2)
            cv2.imshow("Alphabet", alphabet_highlight)
            startPointX += (gridSlotSizeX+1)
            if(top_left[1] == 102):
                if(text[len(text) - 1] == " "):
                    startPointX += 1000
                    continue
                text = text + " "
                continue
            fator = 32
            # print(top_left[1])
            if(top_left[1] < 46):
                fator = 0
            if(top_left[1] > 46 and top_left[1] != 102):
                fator = -33
            text = text + chr(fator + 65 + int((top_left[0] + 9)/19))
            # print(text, top_left, chr(fator + 65 + int((top_left[0] + 9)/19)))
            # os.system('cls') 
            print(text, "\n")
            if a == 0:
                a = waitKey(0)
            else:
                waitKey(3)
        text = text + " "
        startPointY += 43
    print(text)
    exit(0)
