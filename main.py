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

string1 = "droga.png"
string2 = "alfabeto_roboto_mono.png"

im_th, alphabet_th = imgimp.importImages(string1, string2)

#im_th = hough.unrotateImage(im_th)

im_th = op.removeBlackBackgroundWithFindContours(im_th)
cv2.imshow("Black Background Removed", im_th)
characters, boundingBoxes = op.findCharactersUsingContours(np.bitwise_not(im_th))

text = ""
ranger = 5

img2 = im_th.copy()
temp = 0

gridSlotSizeX = 10
gridSlotSizeY = 23
#charactermap = op.buildCharacterMap(im_th)

alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters

for box in boundingBoxes:
    x,y,w,h = box
    if w < 9 or h < 10:
        continue
    startPointX = x - int((gridSlotSizeX - w)/2 + 0.5)
    startPointY = y - int((gridSlotSizeY - h)/2 + 0.5)
    cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
    cv2.imshow("first", img2)
    roi = im_th[y:y+h, x:x+w]
    while(len(img2 > startPointY)):
        startPointX = x - int((gridSlotSizeX - w)/2 + 0.5)
        while(len(img2[0]) > startPointX):
            roi = im_th[startPointY:startPointY + gridSlotSizeY, startPointX:startPointX + gridSlotSizeX]
            cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
            cv2.imshow("passou", img2)

            cv2.waitKey(0)

            result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get location for highest score
            w, h = roi.shape[::-1]  #get shape of roi
            top_left = max_loc  #top left of rectangle
            bottom_right = (top_left[0] + w, top_left[1] + h) #bottom_right of rectangle
            cv2.rectangle(alphabet_highlight, top_left, bottom_right, (0,0,0), 2)
            cv2.imshow("a", alphabet_highlight)
            startPointX += (gridSlotSizeX+1)
            if(top_left[1] == 17):
                if(text[len(text) - 1] == " "):
                    startPointX += 1000
                text = text + " "
                continue
            fator = 0
            if(top_left[1] < 17):
                fator = 32
            text = text + chr(fator + 65 + int((top_left[0] + 9)/19))
            print(text)
        startPointY += 38


    if (temp-x) >= 29:
        text = text + " "
    #print(temp - x, temp, x)
    temp = x
    typei=0
    for i in range(len(charactermap)):
        #print(charactermap[i][0], max_loc[0], charactermap[i][1], max_loc[1], charactermap[i][0], max_loc[0]-ranger, charactermap[i][1], max_loc[1]-ranger, sep=", ")
        if charactermap[i][0] <= max_loc[0]+ranger and charactermap[i][1] <= max_loc[1]+ranger and charactermap[i][0] >= max_loc[0]-ranger and charactermap[i][1] >= max_loc[1]-ranger:
            text = text + str(chr(i+65))
            cv2.circle(alphabet_highlight, max_loc, 2, (255,255,255), 2)
            cv2.imshow("aldskjaçl", alphabet_highlight)
            # cv2.waitKey(0)
            print(text)
        i+=1
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()
