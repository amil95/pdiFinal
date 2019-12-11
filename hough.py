import cv2
import numpy as np
import math
import imageimport as imgimp

def rotateImage(image, angle):
    #print(angle)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255,255,255), flags=cv2.INTER_LINEAR)
    return result
    #borderMode=cv2.BORDER_CONSTANT,
#                           borderValue=(255,255,255)

def unrotateImage(src):
    dst = cv2.Canny(src, 50, 200, None, 3)
    #cv2.imshow("Canny", dst)
    lines = cv2.HoughLines(dst, 1, np.pi / 720, 190, None, 0, 0)
    sum_theta = 0
    #print (lines)
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
    #counts = np.bincount(int(thetas))
    #print (theta)
    average_theta = sum_theta/len(lines)
    #print(average_theta)
    rotated = rotateImage(src, (average_theta*(180/np.pi))-90)
    #cv2.imshow("lines", dst)
    #exit(0)
    return rotated, dst

string1 = "textorotacionado.png"
string2 = "monospaced24.png"

#im_th, alphabet_th = imgimp.importImages(string1, string2)

# cv2.imshow("original", im_th)
# rotated = unrotateImage(im_th)
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)
# exit(0)
