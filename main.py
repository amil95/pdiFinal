import cv2
import numpy as np
import dft as dft
import connectedcomponents as cc
from matplotlib import pyplot as plt
import os
import operator
import math

#find characters using contour
def findCharactersUsingContours(binary_image): # Encontrar cada caracteres utilizando identificacao de blobs por FindContours
    #cv2.imshow("a", binary_image)
    cnts, hier = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cnts = list of contours
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] # bounding Box of each character
    characters = cnts
    #print(characters)
    boundingBoxes = sorted(boundingBoxes, key=operator.itemgetter(0, 1)) # ordena os chars da esquerda para a direita e de cima para baixo
    #for box in boundingBoxes:
        #print(box)
    if __debug__:
        for boundingBox in boundingBoxes:
            x,y,w,h = boundingBox
            #roi = im_th[y:y+h, x:x+w] # the same, for a single character
            # cv2.rectangle(binary_image,(x,y),(x+w,y+h),(255,255,255),2)
            # cv2.imshow("asdf", binary_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
    charactermap = np.zeros((26,2))
    i=0;
    for box in boundingBoxes:
        x,y,w,h = box #the same, for a single character
        roi = alphabet_th[y:y+h, x:x+w] # the same, for a single character
        min_val, max_val, min_loc, max_loc = templateMatching(alphabet_th, roi)
        charactermap[i] = max_loc
        cv2.circle(alphabet_th, max_loc, 2, (255,255,255), 2)
        i+=1
    #cv2.imshow("aksjdfl", alphabet_th)
    return charactermap

#img = cv2.imread("mono.jpg")
img = cv2.imread("multilinemonospaced24.png")
#alphabet = cv2.imread("testimage2.png")
threshold = 200
alphabet = cv2.imread("monospaced24.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grayscale
gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binary image
th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary

### codigo abaixo temporario pra dar bypass em problemas com otsu #######
# gray[gray > 100] = 255
# gray[gray <= 100] = 0
# im_th = gray
# gray_alphabet[gray_alphabet > 100] = 255
# gray_alphabet[gray_alphabet <= 100] = 0
# alphabet_th = gray_alphabet
##########################################################################

# charactermap = buildCharacterMap()
#print(charactermap)

def rotateImage(image, angle):
    #print(angle)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def unrotateImage(src):
    dst = cv2.Canny(src, 50, 200, None, 3)
    cv2.imshow("Canny", dst)
    lines = cv2.HoughLines(dst, 1, np.pi / 360, 190, None, 0, 0)
    sum_theta = 0
    print (lines)
    thetas = np.zeros(len(lines))
    count = 0
    for i in range(len(lines)):
        count += 1
        rho = lines[i][0][0]
        theta = lines [i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(dst, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)
        sum_theta += theta
        thetas[i] = theta
    counts = np.bincount(int(thetas))
    #print (theta)
    average_theta = sum_theta/len(lines)
    #print(average_theta)
    rotated = rotateImage(src, theta*(180/np.pi)+180)
    cv2.imshow("lines", dst)
    cv2.waitKey(0)
    exit(0)
    return rotated

#im_th = unrotateImage(im_th)
characters, boundingBoxes = findCharactersUsingContours(np.bitwise_not(im_th))

text = ""
ranger = 5

img2 = im_th.copy()
temp = 0

gridSlotSizeX = 18
gridSlotSizeY = 23

for box in boundingBoxes:
    x,y,w,h = box
    cv2.circle(img2, (x,y), 2, (255,255,255), 2)
    if w < 4 or h < 5:
        continue
    startPointX = x - int((gridSlotSizeX - w)/2 + 0.5)
    startPointY = y - int((gridSlotSizeY - h)/2 + 0.5)
    roi = im_th[y:y+h, x:x+w]
    cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
    cv2.imshow("first", img2)
    while(len(img2 > startPointY)):
        startPointX = x - int((gridSlotSizeX - w)/2 + 0.5)
        while(len(img2[0]) > startPointX):
            roi = im_th[startPointY:startPointY + gridSlotSizeY, startPointX:startPointX + gridSlotSizeX]
            cv2.rectangle(img2, (startPointX, startPointY),(startPointX + gridSlotSizeX, startPointY + gridSlotSizeY),(0,255,0),2)
            cv2.imshow("first", img2)

            cv2.waitKey(0)

            result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get location for highest score
            w, h = roi.shape[::-1]  #get shape of roi
            top_left = max_loc  #top left of rectangle
            bottom_right = (top_left[0] + w, top_left[1] + h) #bottom_right of rectangle
            alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters
            cv2.rectangle(alphabet_highlight, top_left, bottom_right, (0,0,0), 2)
            cv2.imshow("a", alphabet_highlight)
            startPointX += 19
            if(top_left[1] == 17):
                if(text[len(text) - 1] == " "):
                    startPointX += 1000
                text = text + " "
                continue
            fator = 0
            if(top_left[1] == 2):
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

#
#     meth = "Heisenberg"

    # plt.subplot(121),plt.imshow(result,cmap = 'gray')
    # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(gray_alphabet,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    # plt.suptitle(meth)
    # plt.show()
