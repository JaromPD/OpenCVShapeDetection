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
cv2.createTrackbar("Threshold1", "Parameters", 111, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 149, 255, empty)

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)

    foundRect = False
    foundTri = False

    for cnt in contours:

        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
                foundTri = True
                # Calculate centroid of triangle
                M = cv2.moments(cnt)
                tcx = int(M['m10'] / M['m00'])
                tcy = int(M['m01'] / M['m00'])
                cv2.circle(imgContour, (tcx, tcy), 5, (0, 0, 255), -1)
            elif 7 > objCor >= 4:
                aspRatio = w / float(h)
                objectType = "Rectangle"
                foundRect = True
                # Calculate centroid of rectangle
                rcx = x + w // 2
                rcy = y + h // 2
                cv2.circle(imgContour, (rcx, rcy), 5, (0, 0, 255), -1)
            elif objCor > 7:
                objectType = "Circle"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType,
                        (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)

        if foundRect and foundTri:
            # Draw a line from the centroid of the rectangle to the centroid of the triangle
            cv2.line(imgContour, (rcx, rcy), (tcx, tcy), (0, 0, 255), 5)
            # Label the line with the distance between the centroids
            distance = math.sqrt((rcx - tcx) ** 2 + (rcy - tcy) ** 2)
            cv2.putText(imgContour, str(distance),
                        (rcx - 50, rcy - 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            
            # get dx && dy
            dx = rcx - tcx
            dy = rcy - tcy

            cv2.putText(imgContour, "DX:" + str(dx), (10, frameHeight -  50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(imgContour, "DY:" + str(dy), (10, frameHeight -  30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            if distance <= 110:
                cv2.putText(imgContour, "FOUND", (10, frameHeight - 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
            else:
                cv2.putText(imgContour, "NOT FOUND", (10, frameHeight - 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                if dx > 5:
                    cv2.putText(imgContour, "RIGHT", (10, frameHeight - 90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                elif dx < 5:
                    cv2.putText(imgContour, "LEFT", (10, frameHeight -  90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                if dy > 0:
                    cv2.putText(imgContour, "DOWN", (10, frameHeight -  110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                elif dy < 0:
                    cv2.putText(imgContour, "UP", (10, frameHeight -  110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)

            


while True:
    success, img = cap.read()
    imgContour = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)

    getContours(imgDial, imgContour)

    cv2.imshow("Video", imgGray)
    #cv2.imshow("Result", img)
    #cv2.imshow("Canny", imgCanny)
    cv2.imshow("Dial", imgDial)
    cv2.imshow("Contours", imgContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break