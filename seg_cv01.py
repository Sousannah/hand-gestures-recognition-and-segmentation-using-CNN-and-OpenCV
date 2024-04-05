import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('keras_model_segmentation.h5', "labels_seg.txt")

offset = 20
imgSize = 300
counter=0
pred=0
labels = ['blank', 'fist', 'five', 'ok', 'thumbsdown', 'thumbsup']



while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        modelImg = imgWhite.copy()
        gray = cv2.cvtColor(modelImg, cv2.COLOR_BGR2GRAY)
        gray_inverted = cv2.bitwise_not(gray)
        contours, hierarchy = cv2.findContours(gray_inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, binary = cv2.threshold(gray_inverted, 100, 255, cv2.THRESH_BINARY)

        # Convert binary image to 3-channel grayscale
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # Resize the image to (224, 224)
        resized_img = cv2.resize(binary_rgb, (224, 224))

        prediction, index = classifier.getPrediction(resized_img)
        print(prediction, index)

        pred +=1
        if index == 1:
            counter+=1

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("Segmentation", resized_img)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    # Exit when 'q' is pressed
    if key & 0xFF == ord('q'):
        break

acc = float(counter/pred)
print(acc)
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
