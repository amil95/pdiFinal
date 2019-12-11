import cv2
import numpy as np
import operator

#find characters using contour
def findCharactersUsingContours(binary_image): # Encontrar cada caracteres utilizando identificacao de blobs por FindContours
    cnts, hier = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cnts = list of contours
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] # bounding Box of each character
    characters = cnts
    boundingBoxes = sorted(boundingBoxes, key=operator.itemgetter(0)) # ordena os chars da esquerda para a direita e de cima para baixo
    mins = np.amin(boundingBoxes, axis = 0)
    min_x = mins[0]
    min_y = mins[1]
    cv2.circle(binary_image, (min_x, min_y), 5, (255,255,255), 5)
    # cv2.imshow("asdf", binary_image)
    cv2.waitKey()
    if __debug__:
        for boundingBox in boundingBoxes:
            x,y,w,h = boundingBox
            roi = binary_image[y:y+h, x:x+w] # the same, for a single character
            #cv2.rectangle(binary_image,(min_x,min_y),(x+w,y+h),(255,255,255),2)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    return characters, boundingBoxes, min_x, min_y

def templateMatching(image, template):
    result = cv2.matchTemplate(np.bitwise_not(image), template, cv2.TM_SQDIFF) # template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # get values
    return min_val, max_val, min_loc, max_loc # return values

def buildCharacterMap(alphabet):
    #alphabet = cv2.imread("alphabetALLCAPS.png")
    gray_alphabet = cv2.cvtColor(alphabet, cv2.COLOR_BGR2GRAY) #convert alphabet to grayscale
    th2, alphabet_th = cv2.threshold(gray_alphabet, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary
    characters, boundingBoxes = findCharactersUsingContours(np.bitwise_not(alphabet_th))
    charactermap = np.zeros((26,2))
    i=0
    for box in boundingBoxes:
        x,y,w,h = box #the same, for a single character
        roi = alphabet_th[y:y+h, x:x+w] # the same, for a single character
        min_val, max_val, min_loc, max_loc = templateMatching(alphabet_th, roi)
        charactermap[i] = max_loc
        cv2.circle(alphabet_th, max_loc, 2, (255,255,255), 2)
        i+=1
    #cv2.imshow("aksjdfl", alphabet_th)
    return charactermap

def removeBlackBackgroundWithFindContours(image):
    characters, boundingBoxes = findCharactersUsingContours((image))
    for box in boundingBoxes:
        x,y,w,h = box
        roi = image[y:y+h, x:x+w] # the same, for a single character
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2)
    return roi
