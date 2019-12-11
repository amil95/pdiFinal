import cv2
import numpy as np
import math
import imageimport as imgimp
import statistics

def rotateImage(image, angle):
    #print(angle)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255,255,255), flags=cv2.INTER_LINEAR)
    return result
    #borderMode=cv2.BORDER_CONSTANT,
#                           borderValue=(255,255,255)

def drawLine(img, rho, theta):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    cv2.line(img, pt1, pt2, (255,255,255), 1, cv2.LINE_AA)

def meteHough(src):
    dst = cv2.Canny(src, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, 0.5, np.pi / 720, 140, None, 0, 0)
    lines = sorted(lines, key=lambda x:x[0][0])
    return lines

def getTextLinesFromHough(src):
    lines = meteHough(src)
    rhos = np.zeros(len(lines))
    thetas = np.zeros(len(lines))
    difference_rhos = np.zeros(len(lines))
    copy2 = np.bitwise_not(src.copy())
    for i in range(len(lines)):
        rhos[i] = lines[i][0][0]
        thetas[i] = theta = lines[i][0][1]
        drawLine(copy2, rhos[i], thetas[i])
        if (i > 0):
            difference_rhos[i] = lines[i][0][0] - lines[i-1][0][0]
        else: difference_rhos[i] = 0

    #print (difference_rhos)
    cv2.imshow("Hough Lines after rotation", copy2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    line_count = 0
    text_lines = np.zeros((len(lines),2))
    copy = np.bitwise_not(src.copy())
    slice = 0
    sum_thetas = 0
    count_thetas = 0
    stdev_rhos = statistics.variance(difference_rhos)
    mean_rhos = statistics.mean(difference_rhos)
    #print(np.max(difference_rhos))
    print(stdev_rhos)
    print(mean_rhos)
    print(mean_rhos+stdev_rhos)
    print((difference_rhos))
    for i in range(len(difference_rhos)):
        if (thetas[i]*(180/np.pi) <= 93 and thetas[i]*(180/np.pi) >= 87):
            sum_thetas += thetas[i]
            count_thetas+=1
        if ((difference_rhos[i] > mean_rhos+(stdev_rhos/2) and thetas[i]*(180/np.pi) <= 95 and thetas[i]*(180/np.pi) >= 85)):
            #print(thetas[i]*(180/np.pi))
            text_lines[line_count][0] = (np.sum(rhos[slice:i]))/(i-slice)
            text_lines[line_count][1] = sum_thetas/count_thetas
            #print((np.sum(thetas[slice:i]))/(i-slice)*(180/np.pi), i-slice, sep=' - ')
            drawLine(copy, text_lines[line_count][0], text_lines[line_count][1])
            sum_thetas = 0
            count_thetas = 0
            slice = i
            line_count += 1
    #print(thetas[i]*(180/np.pi))
    text_lines[line_count][0] = (np.sum(rhos[slice:i]))/(i-slice)
    text_lines[line_count][1] = sum_thetas/count_thetas
    #print((np.sum(thetas[slice:i]))/(i-slice)*(180/np.pi), i-slice, sep=' - ')
    drawLine(copy, text_lines[line_count][0], text_lines[line_count][1])
    sum_thetas = 0
    count_thetas = 0
    slice = i
    return copy2, text_lines[:line_count,:-1], copy

def unrotateImage(src):
    lines = meteHough(src)
    #rhos = np.zeros(len(lines))
    #thetas = np.zeros(len(lines))
    #difference_rhos = np.zeros(len(lines))
    sum_thetas = 0
    copy = np.bitwise_not(src)
    for i in range(len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        drawLine(copy, rho, theta)
        #if (i > 0):
            #difference_rhos[i] = lines[i][0][0] - lines[i-1][0][0]
        sum_thetas += theta
    average_theta = sum_thetas/len(lines)
    rotated = rotateImage(src, (average_theta*(180/np.pi))-90)
    # line_count = 0
    # text_lines = np.zeros((len(lines),2))

    # copy = np.bitwise_not(src.copy())

    # slice = 0
    # for i in range(len(difference_rhos)):
    #     if (difference_rhos[i] > 13):
    #         text_lines[line_count][0] = (np.sum(rhos[slice:i]))/(i-slice)
    #         text_lines[line_count][1] = (np.sum(thetas[slice:i]))/(i-slice)
    #         drawLine(copy, text_lines[line_count][0], text_lines[line_count][1])
    #         slice = i
            #line_count += 1
    # cv2.imshow("lines", dst)
    #exit(0)
    return rotated, copy#, text_lines[:line_count,:-1], copy

string1 = "textorotacionado.png"
string2 = "monospaced24.png"

#im_th, alphabet_th = imgimp.importImages(string1, string2)

# cv2.imshow("original", im_th)
# rotated = unrotateImage(im_th)
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)
# exit(0)
