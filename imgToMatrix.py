from copy import deepcopy
import math
import cv2
import numpy as np

def detectLines(gray_img):
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(gray_img, low_threshold, high_threshold)

    lines = cv2.HoughLinesP(edges, 1, np.pi/200, 50, minLineLength=30, maxLineGap=40)

    return lines

def deleteLines(lines, img):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 7)

def detectCircles(img):
    params = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 10
    
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    return detector.detect(img)

def drawCircles(circles, image, outputFile):
    im_with_keypoints = cv2.drawKeypoints(image, circles, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for i in range(len(circles)):
        cv2.putText(im_with_keypoints, str(i),  (int(circles[i].pt[0]), int(circles[i].pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    # Show keypoints
    cv2.imwrite(outputFile, im_with_keypoints)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edgeFromCircle(edge, circle1, threshold):
    # Check if edge connects to a circle
    x1, y1, x2, y2 = edge[0]
    if math.dist((x1, y1), (circle1.pt[0], circle1.pt[1])) < (circle1.size/2 + threshold):
        return True
    if math.dist([x2, y2], [circle1.pt[0], circle1.pt[1]]) < (circle1.size/2 + threshold):
        return True        
    return False

def findNextEdge(edge, edges):
    for nextEdge in edges:
        if edge[0] != nextEdge[0]:
            if edgesLink(edge, nextEdge, 3):
                edges.remove(edge)
                return nextEdge
    return -1

def onSegment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def edgesLink(edge1, edge2, threshold):
    if not checkSlope(edge1, edge2, threshold):
        return False
    x1, y1, x2, y2 = edge1[0]
    x3, y3, x4, y4 = edge2[0]
    if onSegment((x1, y1), (x3, y3), (x2, y2)):
        return True
    if onSegment((x1, y1), (x4, y4), (x2, y2)):
        return True
    return False

def checkSlope(edge1, edge2, threshold):
    x1, y1, x2, y2 = edge1[0]
    x3, y3, x4, y4 = edge2[0]
    if x1 == x2:
        x1 += 0.00000001
    if x3 == x4:
        x3 += 0.00000001
    slope1 = (y2-y1)/(x2-x1)
    slope2 = (y4-y3)/(x4-x3)
    if abs(slope1 - slope2) < threshold:
        return True
    return False

def findLinkedCircle(edge, i, circles, edges):
    for j in range(len(circles)):
        if j != i:
            if edgeFromCircle(edge, circles[j], 10):
                return j
    nextEdge = findNextEdge(edge, edges)
    if nextEdge == -1:
        return -1
    return findLinkedCircle(nextEdge, i, circles, edges)

def createMatrix(circles, edges):
    matrixSize = len(circles)
    matrix = np.zeros((matrixSize, matrixSize))

    # for each edge check if it is between two circles
    for edge in edges:
        for i in range(matrixSize):
            if edgeFromCircle(edge, circles[i],10):
                j = findLinkedCircle(edge, i, circles, edges)
                if j != -1:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
                    break
    return matrix

def main():

    inputFile = "images/patrickGraph.png"
    # Read image
    im = cv2.imread(inputFile, cv2.IMREAD_GRAYSCALE)

    # Detect lines
    graphLinks = detectLines(im)

    # Get array from ndarray
    graphLinks = graphLinks.tolist()

    # Delete lines from image
    deleteLines(graphLinks, im)

    # Write image with deleted lines
    cv2.imwrite('images/patrickGraphTestResult.png', im)

    # Detect circles o new image
    circles = detectCircles(im)

    # Draw detected circles
    circleImage = deepcopy(im)
    drawCircles(circles, circleImage, "images/patrickGraphtestResult.png")

    matrix = createMatrix(circles, graphLinks)
    print(matrix)


if __name__ == "__main__":
    main()