import cv2
import numpy as np
import math

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

cv2.namedWindow("Parameters")

def empty():
    pass

cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, empty)

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Triangle"
                # Calculate centroid of triangle
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(imgContour, (cx, cy), 5, (0, 0, 255), -1)
            elif 7 > objCor >= 4:
                aspRatio = w / float(h)
                objectType = "Rectangle"
                # Calculate centroid of rectangle
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(imgContour, (cx, cy), 5, (0, 0, 255), -1)
            elif objCor > 7:
                objectType = "Circle"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)
            
            return (objCor, cx, cy)

while True:
    success, img = cap.read()
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)

    triangle = None
    square = None

    triangle_data = getContours(imgDial, imgContour)
    if triangle_data and triangle_data[0] == 3:
        triangle = triangle_data[1:]

    square_data = getContours(imgDial, imgContour)
    if square_data and square_data[0] >= 4:
        square = square_data[1:]

    if triangle and square:
        distance = math.sqrt((triangle[0] - square[0])**2 + (triangle[1] - square[1])**2)
        cv2.putText(imgContour, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", imgGray)
    cv2.imshow("Dial", imgDial)
    cv2.imshow("Contours", imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
