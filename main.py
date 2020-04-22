import cv2
import numpy as np
import math

# Global Background
background = None
 
# Region Of Interest
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def euclidean_distances(center_point, other_points):
    '''
        Given a center point and an array of other points, it finds multiple distances from the center to other points.
        @Returns an array of distances
    '''
    distances = []
    for i in range(len(other_points)):
        distances.append(math.sqrt((center_point[0] - other_points[i][0]) ** 2 + (center_point[1] - other_points[i][1]) ** 2))
    return distances

def accumulate(frame, accumulated_weight = 0.5):
    '''
        Given a frame and accumulated weight, compute the weighted average
        @Returns none if there is no backgroud
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
        @Returns thresholded background and countours of the hand
    '''
    global background
    
    # Calculate absolute difference between the background and the passed in frame
    diff = cv2.absdiff(background.astype("uint8"), frame)

    # Apply a threshold to the difference to get the foreground
    ret, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

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
        Then generate a circle with 80% radius of max distance. Cut out obtained thresholded using the generated circle. Then do a bounding box check on remaning contours to determine fingertips
        @Returns finger count and contours of the fingertips
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

    # Draw the circle into circular ROI
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    # Returns the cut out obtained using the mask(circular_roi) on the thresholded hand image.
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    # Grab external contours in circle ROI
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Fingers counted and collect fingertip countours
    count = 0
    fingertip_contours = []
    for contour in contours:
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Increment count of fingers based on two conditions:
        # 1. Contour region is not the very bottom of hand area (the wrist)
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        # 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
        limit_points = ((circumference * 0.25) > contour.shape[0])
            
        if  out_of_wrist and limit_points:
            count += 1
            fingertip_contours.append(contour + (roi_right, roi_top))
            
    return count, fingertip_contours


if __name__ == "__main__":
    # Intialize a video and a frame count
    cam = cv2.VideoCapture(0)
    num_frames = 0

    while True:
        # Read current frame
        ret, frame = cam.read()
        # Flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        # Take the ROI from the frame
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        # Apply grayscale and blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # For the first 60 frames we will calculate the average of the background.
        if num_frames < 60:
            # Accumulate the backgroud
            accumulate(gray)
            if num_frames <= 59:
                cv2.putText(frame_copy, "WAIT! GETTING CURRENT BG", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.imshow("Finger Count", frame_copy)
        else:
            # Segment the hand region if we have the background
            hand = segment(gray)
            if hand is not None:
                thresholded, hand_segment = hand
                # Draw contours around hand segment
                cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0),1)

                # Count the fingers and draw fingertip contours 
                fingers, fingertip_contours = count_fingers(thresholded, hand_segment)
                cv2.drawContours(frame_copy, fingertip_contours , -1, (0, 0, 255),1)
                # Display count
                cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # Display the thresholded image(WINDOW)
                cv2.imshow("Thesholded", thresholded)

        # Draw ROI Rectangle on frame copy
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,255,0), 5)
        # Increment number of frames
        num_frames += 1
        # Display the unthresholded frame with segmented hand(WINDOW)
        cv2.imshow("Finger Count", frame_copy)

        # Close windows with Esc
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Release the camera and destroy all the windows
    cam.release()
    cv2.destroyAllWindows()