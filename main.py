import cv2
import numpy as np
import math, time # Debugging

# This background will be a global variable that we update through a few functions
background = None

# Region Of Interest
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def euclidean_distances(center_point, other_points):
    result = []
    for i in range(len(other_points)):
        result.append(math.sqrt((center_point[0] - other_points[i][0]) ** 2 + (center_point[1] - other_points[i][1]) ** 2))
    return result

def accumulate(frame, accumulated_weight = 0.5):
    '''
        Given a frame and accumulated weight, compute the weighted average
    '''
    global background
    
    # For first time only, create the background from a copy of the frame.
    if background is None:
        background = frame.copy().astype("float")
        return None

    # Compute weighted average then accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment(frame, threshold = 25):
    '''
        Given a frame and threshold, compute countours of foreground and pick the largest area as hand segment
    '''
    global background
    
    # Calculate absolute difference between the backgroud and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the difference to get the foreground
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Grab the external contours from thresholded foreground
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # If length of contours list is not 0, then get the largest external contour area as hand segment
        hand_segment = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment)
    
    return None

def count_fingers(thresholded, hand_segment):
    '''
        Given a thresholded image and hand_segment, compute convex hull then find 4 most outward points. Pick a center of these extreme outward points.
        Generate a circle with 80% radius of max distance.
    '''
    # Compute the convex hull of the hand segment
    conv_hull = cv2.convexHull(hand_segment)

    # Find the most extreme top, bottom, left , right XY coordinates then cast them into tuples.
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    # In theory, the center of the hand is half way between the top and bottom and halfway between left and right
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    # Calculate the Euclidean distances between the assumed center of the hand and the top, left, bottom, and right.
    circle_center = (cX, cY)
    outer_points = [top, left, bottom, right]
    distances = euclidean_distances(circle_center , outer_points)
    max_distance = max(distances)
    
    # Create a circle with radius that is 80% of the max euclidean distance
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    # Not grab an ROI of only that circle
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    # Using bit-wise AND with the cirle ROI as a mask.
    # This then returns the cut out obtained using the mask on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # Grab contours in circle ROI
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Finger count starts at 0
    count = 0

    # loop through the contours to see if we count any more fingers.
    for cnt in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(cnt)

        # Increment count of fingers based on two conditions:
        
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        
        if  out_of_wrist and limit_points:
            count += 1

    return count


if __name__ == "__main__":
    
    cam = cv2.VideoCapture(0)

    # Intialize a frame count
    num_frames = 0

    # keep looping, until interrupted
    while True:
        # get the current frame
        ret, frame = cam.read()

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        frame_copy = frame.copy()

        # Grab the ROI from the frame
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]

        # Apply grayscale and blur to ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # For the first 60 frames we will calculate the average of the background.
        # We will tell the user while this is happening
        if num_frames < 60:
            accumulate(gray)
            if num_frames <= 59:
                cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Finger Count",frame_copy)
                
        else:
            # now that we have the background, we can segment the hand.
            
            # segment the hand region
            hand = segment(gray)

            # First check if we were able to actually detect a hand
            if hand is not None:
                
                # unpack
                thresholded, hand_segment = hand

                # Draw contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

                # Count the fingers
                fingers = count_fingers(thresholded, hand_segment)

                # Display count
                cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # Also display the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # Draw ROI Rectangle on frame copy
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)

        # increment the number of frames for tracking
        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("Finger Count", frame_copy)


        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()