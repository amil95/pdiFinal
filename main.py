import cv2
import numpy as np

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(255*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    firstpx = np.zeros((np.max(labels)+1, 2))
    lastpx = np.zeros((np.max(labels)+1, 2))

    iterator = np.nditer(labels, flags=['multi_index'])
    with iterator:
        while not iterator.finished:
            #print (iterator.multi_index)
            #print (lastpx[iterator[0]][0])
            index = iterator.multi_index
            if(iterator[0] != 0):
                if(firstpx[iterator[0]][0] == 0 and firstpx[iterator[0]][0] == 0):
                    firstpx[iterator[0]] = index
                    #print(iterator[0], firstpx[iterator[0]], sep = ' - ')
                    #print(firstpx)
                #elif(lastpx[iterator[0]][0] == 0 and lastpx[iterator[0]][1] == 0):
                lastpx[iterator[0]] = index
                    #print(index[0], index[1], sep=',')
                    #print(lastpx[iterator[0]][1])
                #elif(index[1] >= lastpx[iterator[0]][1]):
                    #lastpx[iterator[0]][0] = 30

                #if(index[0] > lastpx[iterator[0]][0]):
                #    lastpx[iterator[0]][0] = index[0]
                #and index[1] > lastpx[iterator[0]][1]
            iterator.iternext()
    #print (np.max(firstpx))
    #print(firstpx)
    #print(lastpx)

    for component in np.nditer(labels):
        #print (firstpx[component][0])
        #cv2.rectangle(labeled_img, (2, 5), (5, 30), (255,255,255), 2)
        #cv2.circle(labeled_img, (int(firstpx[component][1]),int(firstpx[component][0])), 1, (255,255,255), 1)
        cv2.rectangle(labeled_img, (int(firstpx[component][1]),int(firstpx[component][0])), (int(lastpx[component][1]),int(lastpx[component][0])), (255,255,255), 1)

#             with it:
# ...        while not it.finished:
# ...            addop(it[0], it[1], out=it[2])
# ...            it.iternext()
# ...        return it.operands[2]


    #cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    cv2.imshow('labeled.png', labeled_img)

img = cv2.imread("umalinha.png")
threshold = 200

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary
im_th_inv = np.bitwise_not(im_th) #invert binary

ret, labels = cv2.connectedComponents(im_th, connectivity = 8)
print(ret)

components = np.array(labels, dtype=np.uint8)

imshow_components(labels)

'''h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)'''
im_floodfill = im_th_inv.copy()
cv2.floodFill(im_floodfill, None, (0,0), 255);
im_floodfill = np.bitwise_not(im_floodfill)

cv2.imshow("binary", im_th_inv)
cv2.imshow("floodfill", im_floodfill)
cv2.imshow("connectedcomponents", components)
cv2.waitKey(0)

#im_floodfill = im_th.copy()

#h, w = im_th.shape[:2]
#mask = np.zeros((h + 2, w + 2), np.uint8)

#cv2.floodFill(im_floodfill, mask, (0, 0), 255)

#im_floodfill_inv = cv2.bitwise_not(im_floodfill)

#fill_image = im_th | im_floodfill_inv

cv2.destroyAllWindows()
