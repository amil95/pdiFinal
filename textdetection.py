import cv2
import numpy as np
import operations as op
import statistics

def waitKey(b):
    a = cv2.waitKey(b)
    if a == 49:
        exit(0)
    if a == 50:
        return 1
    else:
        return 0

# def findCharacterWidth(boundingBoxes, text_lines, img2):

#     boundingBoxes_a = np.asarray(boundingBoxes)

#     stdev_w = statistics.stdev(np.squeeze(boundingBoxes_a[:,2:3]))
#     stdev_h = statistics.stdev(np.squeeze(boundingBoxes_a[:,3:4]))

#     meanW = statistics.mean(np.squeeze(boundingBoxes_a[:,2:3]))
#     meanH = statistics.mean(np.squeeze(boundingBoxes_a[:,3:4]))

#     total = 0
#     quant = 0
#     xant = 0
#     firstx = 1000
#     lastx = 0
#     print(text_lines)
#     for box in boundingBoxes:
#         x,y,w,h = box
#         if(y < text_lines[0] and w > meanW - stdev_w and h > meanH - stdev_h):
#             quant += 1
#             if(firstx > x):
#                 firstx = x
#             if(lastx < x):
#                 lastx = x
#             if(x - xant > 2*w):
#                 quant += int((x-xant)/(2*w) + 0.5)
#             xant = x
#             cv2.rectangle(img2, (x,y),(x+w, y+h),(0,255,0),2)
#             cv2.imshow("media", img2)
#             waitKey(1)
#             #print(int(((lastx+w)-firstx)/quant + 0.5))
#     characterWidth = (int(((lastx+w)-firstx)/quant + 0.5))
    # return 20
    
# print("Character Width: " , findCharacterWidth(boundingBoxes, text_lines, img2))
# gridSlotSizeX = int(((lastx+w)-firstx)/quant + 0.5)
def detectText(im_th, text_lines):

    characters, boundingBoxes, min_x, min_y = op.findCharactersUsingContours(np.bitwise_not(im_th))

    boundingBoxes_a = np.asarray(boundingBoxes)

    stdev_w = statistics.stdev(np.squeeze(boundingBoxes_a[:,2:3]))
    stdev_h = statistics.stdev(np.squeeze(boundingBoxes_a[:,3:4]))

    meanW = statistics.mean(np.squeeze(boundingBoxes_a[:,2:3]))
    meanH = statistics.mean(np.squeeze(boundingBoxes_a[:,3:4]))

    total = 0
    quant = 0
    xant = 0
    firstx = 1000
    lastx = 0
    img2 = im_th.copy()
    for box in boundingBoxes:
        x,y,w,h = box
        cv2.rectangle(img2, (x,y),(x+w, y+h),(0,255,0),2)
        cv2.imshow("menes1", img2)
        waitKey(0)
        if(y < text_lines[0] and w > meanW - stdev_w and h > meanH - stdev_h):
            quant += 1
            if(firstx > x):
                firstx = x
            if(lastx < x):
                lastx = x
            if(x - xant > 2*w):
                quant += int((x-xant)/(2*w) + 0.5)
            xant = x
            cv2.rectangle(img2, (x,y),(x+w, y+h),(0,255,0),2)
            cv2.imshow("media", img2)
            waitKey(1)
            #print(int(((lastx+w)-firstx)/quant + 0.5))
    characterWidth = (int(((lastx+w)-firstx)/quant + 0.5))

    text = ""
    ranger = 5

    # temp = 0
    boundingBoxes_a = np.asarray(boundingBoxes)

    stdev_w = statistics.stdev(np.squeeze(boundingBoxes_a[:,2:3]))
    stdev_h = statistics.stdev(np.squeeze(boundingBoxes_a[:,3:4]))

    meanW = statistics.mean(np.squeeze(boundingBoxes_a[:,2:3]))
    meanH = statistics.mean(np.squeeze(boundingBoxes_a[:,3:4]))


    gridSlotSizeX = 18
    gridSlotSizeY = 31
    #charactermap = op.buildCharacterMap(im_th)
    a = 0
    # line = np.zeros((len(text_lines),100))
        
    W = 1000.
    oriimg = im_th.copy()
    height, width = oriimg.shape
    imgScale = 18/characterWidth
    newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    #img2 = cv2.resize(oriimg,(int(newX),int(newY)))

    characters, boundingBoxes, min_x, min_y = op.findCharactersUsingContours(np.bitwise_not(img2))

    for box in boundingBoxes:
        x,y,w,h = box
        cv2.rectangle(img2, (x,y),(x+w, y+h),(0,255,0),2)
        cv2.imshow("menes", img2)
        waitKey(0)
        continue

        if w < meanW - stdev_w and h < meanH - stdev_h:
            continue

        startPointX = min_x - int((gridSlotSizeX - w)/2 + 0.5)
        startPointY = min_y - int((gridSlotSizeY - h)/2 + 0.5) + 4
        
        i = 0
        waitKey(0)
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
                startPointX += (gridSlotSizeX + 1)
                if(top_left[1] == 102):
                    if(text[len(text) - 1] == " "):
                        startPointX += 1000
                        continue
                    text = text + " "
                    continue
                fator = 32
                if(top_left[1] < 35):
                    fator = 0
                if(top_left[1] > 60 and top_left[1] != 102):
                    fator = -33
                text = text + chr(fator + 65 + int((top_left[0] + 9)/19))
                print(text, "\n")
                if a == 0:
                    a = waitKey(0)
                else:
                    waitKey(3)
            text = text + " "
            i += 1
        print(text)
    exit(0)