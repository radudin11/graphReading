import numpy as np
import cv2
import random


def deleteLines(edgeLines, img):
    for line1 in edgeLines:
        x1, y1, x2, y2 = line1[0]

            


if __name__ == '__main__':
    img = cv2.imread('images/graphTest.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    low_threshold = 10
    high_threshold = 200
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    line_img = np.copy(img)*0 # creating a blank to draw lines on

    edgeLines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    # deleteLines(edgeLines, img)
    for line in edgeLines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (random.randint(0, 255), random.randint(0,255), random.randint(0, 255)), 2)
    
    cv2.drawMarker(img, (590, 294), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    cv2.drawMarker(img, (733, 208), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    

    cv2.imshow('img', img)
    cv2.imwrite('images/graphTestResult.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()