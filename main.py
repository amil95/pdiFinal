#python modules
import cv2
import numpy as np
# import dft as dft
import connectedcomponents as cc
from matplotlib import pyplot as plt
import os
#custom modules
import math
import hough
import imageimport as imgimp
import operations as op
import operator
import statistics

def waitKey(b):
    a = cv2.waitKey(b)
    if a == 49:
        exit(0)
    if a == 50:
        return 1
    else:
        return 0

string1 = "texto_roboto_mono_24_rotated.png"
# string1 = "lorem_roboto_mono_20.png"
# string1 = "resizeimg.jpg"
string2 = "alfabeto_roboto_mono.png"

original, im_th, alphabet_th = imgimp.importImages(string1, string2)

cv2.imshow("Original", original)
waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Binary", im_th)
waitKey(0)
cv2.destroyAllWindows()

im_th, im_lines = hough.unrotateImage(im_th)

cv2.imshow("Hough Lines", im_lines)
waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Correct rotation", im_th)
waitKey(0)
cv2.destroyAllWindows()

im_lines_rotated, text_lines, im_aggregated = hough.getTextLinesFromHough(im_th)

cv2.imshow("Hough Lines rotated", im_lines_rotated)
waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Hough Lines aggregated", im_aggregated)
waitKey(0)
cv2.destroyAllWindows()


characters, boundingBoxes, min_x, min_y = op.findCharactersUsingContours(np.bitwise_not(im_th))

text = ""
ranger = 5

img2 = im_th.copy()
temp = 0

boundingBoxes_a = np.asarray(boundingBoxes)

stdev_w = statistics.stdev(np.squeeze(boundingBoxes_a[:,2:3]))
stdev_h = statistics.stdev(np.squeeze(boundingBoxes_a[:,3:4]))

meanW = statistics.mean(np.squeeze(boundingBoxes_a[:,2:3]))
meanH = statistics.mean(np.squeeze(boundingBoxes_a[:,3:4]))

#charactermap = op.buildCharacterMap(im_th)
a = 0
# line = np.zeros((len(text_lines),100))
    
def findCharacterWidth(boundingBoxes, text_lines, img2):
    total = 0
    quant = 0
    xant = 0
    firstx = 1000
    lastx = 0
    for box in boundingBoxes:
        x,y,w,h = box
        if( y < text_lines[0] and w > meanW - stdev_w and h > meanH - stdev_h):
            quant += 1
            if(firstx > x):
                firstx = x
            if(lastx < x):
                lastx = x
            if(x - xant > 2*w):
                quant += int((x-xant)/(2*w) + 0.5)
            xant = x
            # cv2.rectangle(img2, (x,y),(x+w, y+h),(0,255,0),2)
            # cv2.imshow("media", img2)
            # waitKey(0)
    return(int(((lastx+w)-firstx)/quant + 0.5))
chrWidth = findCharacterWidth(boundingBoxes, text_lines, img2)
print("Character Width: " , chrWidth)
gridSlotSizeX = chrWidth + (1 - chrWidth%2)
gridSlotSizeY = int(chrWidth *1.8)
# gridSlotSizeX = int(((lastx+w)-firstx)/quant + 0.5)
if(chrWidth < 18):
    oriimg = alphabet_th
    height, width = oriimg.shape
    imgScale = chrWidth/18
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    alphabet_th = cv2.resize(oriimg,(int(newX),int(newY)))
    cv2.imshow("Show by CV2",alphabet_th)
    cv2.waitKey(0)
    # cv2.imwrite("resizeimg.jpg",newimg)

for box in boundingBoxes:
    x,y,w,h = box
    if w < meanW - stdev_w or h < meanH - stdev_h:
        continue

    startPointX = min_x - int((chrWidth - w)/2 + 0.5)
    startPointY = min_y - int((gridSlotSizeY - h)/2 + 0.5) + 4
    
    i = 0
    roi = im_th[y:y+h, x:x+w]
    while(i < len(text_lines)):
        startPointY = int(text_lines[i] - (gridSlotSizeY/2))
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
            startPointX += (gridSlotSizeX)
            fator = 32
            if(top_left[1] < len(alphabet_th)/4):
                fator = 0
            if(top_left[1] > 60 and not (top_left[1] >= 100 and top_left[1] < 120)):
                fator = -33
            if((fator + 65 + int((top_left[0] + (w/2))/(gridSlotSizeX))) == 32 and text[len(text) - 1] == " "):
                startPointX += 1000
                continue
            text = text + chr(fator + 65 + int((top_left[0] + (w/2))/(gridSlotSizeX)))
            print(text, "\n")
            if a == 0:
                a = waitKey(0)
            else:
                waitKey(3)
        text = text + " "
        i += 1
    print(text)
    exit(0)
