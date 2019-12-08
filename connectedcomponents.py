import cv2
import numpy as np

def findCharactersUsingConnectedComponents(image):
    # Label connected components
    ret, image = cv2.connectedComponents(image, connectivity = 8)
    # Map component image to hue val in a black hsv image
    label_hue = np.uint8(255*image/np.max(image))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    firstpx = np.zeros((np.max(image)+1, 2))
    lastpx = np.zeros((np.max(image)+1, 2))

    iterator = np.nditer(image, flags=['multi_index'])
    with iterator:
        while not iterator.finished:
            index = iterator.multi_index
            if(iterator[0] != 0):
                if(firstpx[iterator[0]][0] == 0 and firstpx[iterator[0]][0] == 0):
                    firstpx[iterator[0]] = index
                lastpx[iterator[0]] = index
            iterator.iternext()
    for component in np.nditer(image):
        cv2.rectangle(labeled_img, (int(firstpx[component][1]),int(firstpx[component][0])), (int(lastpx[component][1]),int(lastpx[component][0])), (255,255,255), 1)

    #rectangle = np.zeros((int(lastpx[1][0]-firstpx[1][0]),int(lastpx[1][1]-firstpx[1][1])))
    return labeled_img
