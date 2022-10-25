import numpy as np
import cv2

if __name__ == '__main__':
    image = cv2.imread('images/lineDetectionTestResult.png')
    
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    params.filterByArea = False
    params.minArea = 100
    
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
    keypoints = detector.detect(image)
    
    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])
        radius = int(keypoint.size / 2)

        blobs = cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        blobs = cv2.drawMarker(image, (x + radius, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    number_of_blobs = len(keypoints)
    # print(keypoints[0].pt, keypoints[0].size)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    
    # Show blobs
    cv2.imwrite('images/lineDetectionTestResult.png', blobs)
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()