import cv2
import numpy as np

# Find character using connected components
def findCharactersUsingConnectedComponents(image):
    # Label connected components
    ret, image = cv2.connectedComponents(im_th, connectivity = 8)
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

#find characters using contour
def findCharactersUsingContours(binary_image):
    ctrs, hier = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    #cv2.imshow("contours", binary_image)

    for i, ctr in enumerate(sorted_ctrs):
        x,y,w,h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = binary_image[y:y+h, x:x+w]
        # show ROI
        cv2.imshow('segment no:'+str(i),roi)
        #cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)

def getDFT(binary_image):
    rows,cols = binary_image.shape
    m = cv2.getOptimalDFTSize(rows)
    n = cv2.getOptimalDFTSize(cols)
    padded_image = cv2.copyMakeBorder(binary_image, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0,0,0])
    #cv2.imshow("padded", padded_image)

    planes = [np.float32(padded_image), np.zeros(padded_image.shape, np.float32)]
    complexI = cv2.merge(planes)         # Add to the expanded another plane with zeros
    cv2.dft(complexI, complexI)
    cv2.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv2.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv2.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv2.log(magI, magI)

    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp

    cv2.normalize(magI, magI, 0, 1, cv2.NORM_MINMAX) # Transform the matrix with float values into a
    #cv2.imshow("fourier", magI)
    return magI




img = cv2.imread("umalinharotacionado.png")
threshold = 200

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
th, im_th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #ensure binary
im_th_inv = np.bitwise_not(im_th) #invert binary


labeled_img = findCharactersUsingConnectedComponents(im_th)
contours = findCharactersUsingContours(im_th)
dft = getDFT(im_th)
#dft = cv2.fftshift(dft)
#edge = cv2.Canny(dft,100,200)
#imshow("edge", edge)
'''h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)'''
im_floodfill = im_th_inv.copy()
cv2.floodFill(im_floodfill, None, (0,0), 255);
im_floodfill = np.bitwise_not(im_floodfill)

cv2.imshow("Connected Components", labeled_img)
#cv2.imshow("Contours", contours)
cv2.imshow("DFT Magnitude", dft)
cv2.waitKey(0)

#im_floodfill = im_th.copy()

#h, w = im_th.shape[:2]
#mask = np.zeros((h + 2, w + 2), np.uint8)

#cv2.floodFill(im_floodfill, mask, (0, 0), 255)

#im_floodfill_inv = cv2.bitwise_not(im_floodfill)

#fill_image = im_th | im_floodfill_inv

cv2.destroyAllWindows()
