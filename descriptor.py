import cv2

global currFrame, lastFrame
source = "210331_01_Oxford City_4k_050_preview.webm"
cap = cv2.VideoCapture(source)
ret, currFrame = cap.read()

while True:
    ret, frame = cap.read()
    lastFrame = currFrame
    currFrame = frame

    #convert to grayscale
    lastGrayFr = cv2.cvtColor(lastFrame,cv2.COLOR_BGR2GRAY)
    currGrayFr = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)

    #initialize ORB detector
    orb = cv2.ORB_create()

    #detect keypoints and descriptors
    im1Kps, im1Desc = orb.detectAndCompute(lastGrayFr,None)
    im2Kps, im2Desc = orb.detectAndCompute(currGrayFr,None)

    #matching
    matcher = cv2.BFMatcher()
    matches = matcher.match(im1Desc,im2Desc)

    final_img = cv2.drawMatches(lastFrame,im1Kps,currFrame,im2Kps, matches[:20],None)
    final_img = cv2.resize(final_img, (1000,650))
    
    # Show the final image
    cv2.imshow("Matches", final_img)
    cv2.waitKey(3000)

    #cv2.imshow('windowName',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()

