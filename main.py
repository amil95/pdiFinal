import cv2
import numpy as np
import dft as dft
import connectedcomponents as cc
from matplotlib import pyplot as plt
import os

#find characters using contour
def findCharactersUsingContours(binary_image):
    cnts, hier = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # characters = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1][1], reverse=False)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    characters, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][1], reverse=True))
	# (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][0], reverse=reverse))
    return characters

def buildCharacterMap():
    alphabet = cv2.imread("alphabetALLCAPS.png")
    gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
    th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary
    characters = findCharactersUsingContours(np.bitwise_not(alphabet_th))
    charactermap = np.zeros((26,2,2))
    for i, character in enumerate(characters):
        x,y,w,h = cv2.boundingRect(character) #the same, for a single character
        roi = alphabet_th[y:y+h, x:x+w] # the same, for a single character
        # filename = "/charactermaps/"+str(chr(i+65))+".png"
        # filename = str(filename)
        # cv2.imwrite(os.path.join(filename), roi)
        #cv2.imshow(str(i+65), roi) # show character
        result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        charactermap[i][0] = max_loc
        charactermap[i][1] = min_loc
    #print(charactermap)
    return charactermap



img = cv2.imread("testessss.png")
alphabet = cv2.imread("alphabetALLCAPS.png")
threshold = 200

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grayscale
gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binary image
#im_th_inv = np.bitwise_not(im_th) #invert binary
th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary


#labeled_img = cc.findCharactersUsingConnectedComponents(im_th.copy()) #find blobs with connectedComponents
#dft = dft.getDFT(im_th.copy()) #Get dft
#im_floodfill = im_th_inv.copy() # invert image for floodfill
#cv2.floodFill(im_floodfill, None, (0,0), 255); # find blobs with floodfill
#im_floodfill = np.bitwise_not(im_floodfill) # invert image back to original

#cv2.imshow("a", im_th)
charactermap = buildCharacterMap()
characters = findCharactersUsingContours(np.bitwise_not(im_th))

#for i, ctr in enumerate(characters): # loop found characters
     #x,y,w,h = cv2.boundingRect(ctr) # get bounds for each character
     # Getting ROI
     #roi = im_th[y:y+h, x:x+w] # get region of interest for each character
     # show ROI
     #cv2.imshow("segment", roi)

text = ""
ranger = 5
#character = findCharactersUsingContours(np.bitwise_not(im_th)) #find blobs with Contours
img2 = im_th.copy()
temp = 0
for i, character in enumerate(characters):
    x,y,w,h = cv2.boundingRect(character) #the same, for a single character
    roi = im_th[y:y+h, x:x+w] # the same, for a single character
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    result = cv2.matchTemplate(np.bitwise_not(alphabet_th), roi, cv2.TM_SQDIFF) # template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get location for highest score
    w, h = roi.shape[::-1]  #get shape of roi
    top_left = min_loc  #top left of rectangle
    bottom_right = (top_left[0] + w, top_left[1] + h) #bottom_right of rectangle
    alphabet_highlight = alphabet_th.copy() # copy original image for highlight characters
    cv2.rectangle(alphabet_highlight, top_left, bottom_right, (0,0,0), 2)
    # cv2.imshow("result", result)
    # cv2.imshow("highlight", alphabet_highlight)
    # cv2.imshow("asdlf", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if (temp-x) >= 29:
        text = text + " "
    #print(temp - x, temp, x)
    temp = x
    for i in range(len(charactermap)):
        if charactermap[i][0][0] <= max_loc[0]+ranger and charactermap[i][0][1] <= max_loc[1]+ranger and charactermap[i][0][0] >= max_loc[0]-ranger and charactermap[i][0][1] >= max_loc[1]-ranger:
            # print (charactermap[i][0][0], max_loc[0], sep = " - ")
            # print (charactermap[i][0][1], max_loc[1], sep = " - ")
            text = text + str(chr(i+65))

    #print(text)
print(text[::-1])
#cv2.imshow('segment no:', img2) # show character
#cv2.imshow("roi", alphabet_highlight)
# cv2.imshow("img2", img2)
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
