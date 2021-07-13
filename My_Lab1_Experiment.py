import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.utils import plot_model
import math
from matplotlib import pyplot as plt

def preprocess(imgName):
    imgGray = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)

    ret, imgBin = cv2.threshold(imgGray, 40, 255, cv2.THRESH_BINARY_INV)

    print('Initial binary representation')
    sf = 30
    cv2_imshow(cv2.resize(imgBin, (int(imgBin.shape[1]*sf/100), int(imgBin.shape[0]*sf/100)), interpolation = cv2.INTER_AREA))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    imgDst = cv2.morphologyEx(imgBin, cv2.MORPH_CLOSE, kernel)
    imgDst = cv2.erode(imgDst, None, iterations = 1)
    imgDst = cv2.dilate(imgDst, None, iterations = 1)

    contours, hierarchy = cv2.findContours(imgDst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy, imgDst


def removeInvalidContours(imgBin, contours, areaMin, areaMax):
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < areaMin or area > areaMax:
            m = cv2.moments(contour)
            cx = int(m['m10']/m['m00'])
            cy = int(m['m01']/m['m00'])
            seedPoint = (cx, cy)  # contour[0]
            cv2.floodFill(imgBin, None, seedPoint, 0) # flags = 4|(255<<8))
    print('After preprocessing')
    sf = 30
    cv2_imshow(cv2.resize(imgBin, (int(imgBin.shape[1]*sf/100), int(imgBin.shape[0]*sf/100)), interpolation = cv2.INTER_AREA)) 


def borderContrast(imgBin, imgGray, contour):
    kernel = np.ones((5, 5), np.uint8)
    
    imgBrdOut = cv2.morphologyEx(imgBin, cv2.MORPH_DILATE, kernel, iterations=1)
    imgBrdOut = cv2.morphologyEx(imgBrdOut, cv2.MORPH_GRADIENT, kernel, iterations=1)
    cv2.imwrite('sample_data/imgBrdOut.png', imgBrdOut)

    imgBrdIn = cv2.morphologyEx(imgBin, cv2.MORPH_ERODE, kernel, iterations=1)
    imgBrdIn = cv2.morphologyEx(imgBrdIn, cv2.MORPH_GRADIENT, kernel, iterations=1)
    cv2.imwrite('sample_data/imgBrdIn.png', imgBrdIn)

    meanBrdOut, stddevBrdOut = cv2.meanStdDev(imgGray, mask=imgBrdOut)
    meanBrdIn, stddevBrdIn = cv2.meanStdDev(imgGray, mask=imgBrdIn)

    if meanBrdIn[0][0] > meanBrdOut[0][0]:
        return abs(meanBrdIn[0][0] - meanBrdOut[0][0]) / (meanBrdIn[0][0]+1)
    else:
        return abs(meanBrdIn[0][0] - meanBrdOut[0][0]) / (meanBrdOut[0][0]+1)


def elongation(contour):
    m = cv2.moments(contour)
    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])

    max = 0
    min = 10000000000000000000
    for kp in contour:
        r = math.sqrt((kp[0][0] - cx) ** 2 + (kp[0][1] - cy) ** 2)
        if (r >= max):
            max = r
        if (r <= min):
            min = r
    fi = math.sqrt(1 - min ** 2 / max ** 2)
    return fi

def contoursmoothness(contour):
    kpCnt = len(contour)
    x = 0
    y = 0

    for kp in contour:
        x = x+kp[0][0]
        y = y+kp[0][1]
    x = x / kpCnt
    y = y / kpCnt
    
    sum_r = 0
    for kp in contour:
        sum_r += math.sqrt((kp[0][0] - x) ** 2 + (kp[0][1] - y) ** 2)
    m = sum_r / kpCnt

    sum_rm = 0
    for kp in contour:
        sum_rm += (math.sqrt((kp[0][0] - x) ** 2 + (kp[0][1] - y) ** 2) - m) ** 2
    fi = math.sqrt(sum_rm / kpCnt)

    return fi

def calculateFeatures(imgName):
    imgGray = cv2.imread(imgName)

    contours, hierarchy, imgBin = preprocess(imgName)

    removeInvalidContours(imgBin, contours, 50, 1000)

    contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    delta = 5
    H, W, channels = imgGray.shape
    features = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        if y > delta and y < H-2*delta-1 and x > delta and x < W-2*delta-1:
            imgGrayRoi = imgGray[y-delta:y+h+2*delta, x-delta:x+w+2*delta]
            imgBinRoi = imgBin[y-delta:y+h+2*delta, x-delta:x+w+2*delta]
            features.append([borderContrast(imgBinRoi, imgGrayRoi, contour),
                            elongation(contour),
                            contoursmoothness(contour)])
    return features


def testClassifier(modelName, imgName):
    model = load_model(modelName)
    imgGray = cv2.imread(imgName)
    imgMarked = imgGray

    contours, hierarchy, imgBin = preprocess(imgName)
    removeInvalidContours(imgBin, contours, 50, 1000)
    contours, hierarchy = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    delta = 5
    H, W, channels = imgGray.shape
    contInd = 0

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        if y > delta and y < H-2*delta-1 and x > delta and x < W-2*delta-1:
            imgGrayRoi = imgGray[y-delta:y+h+2*delta, x-delta:x+w+2*delta]
            imgBinRoi = imgBin[y-delta:y+h+2*delta, x-delta:x+w+2*delta]
            
            feature = [borderContrast(imgBinRoi, imgGrayRoi, contour),
                       elongation(contour),
                       contoursmoothness(contour)]
            res = model.predict(np.array([feature]))
            
            if res[0,0] > res[0,1]:
                cv2.drawContours(imgMarked, contours, contInd, color=(0,255,0), thickness = 1, lineType = 0)
            else:
                cv2.drawContours(imgMarked, contours, contInd, color=(0,0,255), thickness = 1, lineType = 0)
            contInd += 1

    print('Marked')
    cv2_imshow(imgMarked)


def trainClassifier(modelName, features1, features2):
    train = []
    label = []
    label1 = [1, 0]
    label2 = [0, 1]

    npa1 = np.array(features1)
    npa2 = np.array(features2)

    for i in range(len(npa1[:,0])):
        train.append(npa1[i,:])
        label.append(label1)

    for i in range(len(npa2[:,0])):
        train.append(npa2[i,:])
        label.append(label2)

    train = np.array(train)
    label = np.array(label)
    rng_state = np.random.get_state()
    np.random.shuffle(train)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

    model = Sequential()
    model.add(Dense(7, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    print(model.summary())
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    model.fit(train, label, epochs=100, verbose=1)
    
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    model.save(modelName)
    print(model.get_weights())


def displayFeatures(features1, features2):
    plt.subplot(2, 2, 1)
    plt.title('BorderContrast')
    npa1 = np.array(features1)
    npa2 = np.array(features2)
    args = dict(histtype='stepfilled', alpha=0.7, bins=50)
    plt.hist(npa1[:,0], **args)
    plt.hist(npa2[:,0], **args)
    
    plt.subplot(2, 2, 2)
    plt.title('Elongation')
    plt.hist(npa1[:,1], **args)
    plt.hist(npa2[:,1], **args)
    
    plt.subplot(2, 2, 3)
    plt.title('Contoursmoothness')
    plt.hist(npa1[:,2], **args)
    plt.hist(npa2[:,2], **args)
    plt.show()


features1 = []
features2 = []
features1 = calculateFeatures('semechki2.png')
features2 = calculateFeatures('soya1.png')
displayFeatures(features1, features2)

modelName = 'classifier.h5'
trainClassifier(modelName, features1, features2)

testClassifier(modelName, 'smes2.png')


