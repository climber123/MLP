import cv2
from google.colab.patches import cv2_imshow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def removeInvalidContours(imgBin, contours, areaMin, areaMax):
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < areaMin or area > areaMax:
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            seedPoint = (cx, cy)  # contour[0]
            cv2.floodFill(imgBin, None, seedPoint, 0)
    print('After preprocessing')
    sf = 30
    cv2_imshow(cv2.resize(imgBin, (int(imgBin.shape[1]*sf/100), int(imgBin.shape[0]*sf/100)), interpolation = cv2.INTER_AREA))

image1 = cv2.imread("semechki2.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("soya1.png", cv2.IMREAD_GRAYSCALE)
ret1,binar1=cv2.threshold(image1,40,255,cv2.THRESH_BINARY_INV)
ret2,binar2=cv2.threshold(image2,40,255,cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed1 = cv2.morphologyEx(binar1, cv2.MORPH_CLOSE, kernel)
closed2 = cv2.morphologyEx(binar2, cv2.MORPH_CLOSE, kernel)
closed1 = cv2.erode(closed1, None, iterations = 2)
closed1 = cv2.dilate(closed1, None, iterations = 2)
closed2 = cv2.erode(closed2, None, iterations = 2)
closed2 = cv2.dilate(closed2, None, iterations = 2)

contours1, hierarchy1=cv2.findContours(closed1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy2=cv2.findContours(closed2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours( image1, contours1, -1, (255,0,0), 1, cv2.LINE_AA, hierarchy1, 1 )
#cv2.drawContours( image2, contours2, -1, (255,0,0), 1, cv2.LINE_AA, hierarchy2, 1 )

removeInvalidContours(closed1, contours1, 50, 1000)
removeInvalidContours(closed2, contours2, 50, 1000)

#cv2_imshow(binar1)
#print('----------------------------')
#cv2_imshow(binar2)
#print('----------------------------')
#cv2_imshow(image1)
#print('----------------------------')
#cv2_imshow(image2)
#print('----------------------------')
#cv2_imshow(closed1)
#cv2_imshow(closed2)