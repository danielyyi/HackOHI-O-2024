import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

def calculate_angle(a, b, c):
    """Calculate the angle between three points (a, b, c)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Prevent NaN errors
    return np.degrees(angle)

def palm_orientation(wrist, thumb):
    """Determine palm orientation based on thumb position."""
    # Compare x-coordinates of the wrist and thumb
    if thumb[0] > wrist[0]:  # Thumb is on the right side of the wrist
        return "Palm Up"
    else:  # Thumb is on the left side of the wrist
        return "Palm Down"

# Open a video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Process the frame for pose detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)  # Flip the image for a mirror effect
        pose_results = pose.process(image)
        hands_results = hands.process(image)
        
        # Convert the image back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # Function to process one arm
            def process_arm(shoulder_landmark, elbow_landmark, wrist_landmark, side):
                shoulder = [landmarks[shoulder_landmark].x, landmarks[shoulder_landmark].y]
                elbow = [landmarks[elbow_landmark].x, landmarks[elbow_landmark].y]
                wrist = [landmarks[wrist_landmark].x, landmarks[wrist_landmark].y]

                # Calculate the angle for the arm
                angle = calculate_angle(shoulder, elbow, wrist)

                # Adjust angles for flipped image
                if side == "left":
                    angle = 180 - angle  # Flip angle for left arm
                elif side == "right":
                    angle = 180 - angle  # Flip angle for right arm

                # Initialize hand orientation
                palm_orientation_result = "Unknown"
                phone_detected = False  # Initialize phone detected status

                # Draw the landmarks for the arm
                for landmark in [shoulder_landmark, elbow_landmark, wrist_landmark]:
                    x = int(landmarks[landmark].x * frame.shape[1])
                    y = int(landmarks[landmark].y * frame.shape[0])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                # Draw lines between the points for the arm
                cv2.line(image, (int(shoulder[0] * frame.shape[1]), int(shoulder[1] * frame.shape[0])),
                         (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])), (255, 0, 0), 2)
                cv2.line(image, (int(elbow[0] * frame.shape[1]), int(elbow[1] * frame.shape[0])),
                         (int(wrist[0] * frame.shape[1]), int(wrist[1] * frame.shape[0])), (255, 0, 0), 2)

                # Process hand landmarks if present
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        # Check which hand it is
                        hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].x
                        hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].y

                        if side == "right" and hand_x > wrist[0]:  # Right hand
                            index_finger_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y]
                            thumb = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP.value].x,
                                     hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP.value].y]

                            # Determine palm orientation using the improved method
                            palm_orientation_result = palm_orientation(wrist, thumb)

                            # Check if angle is in range and palm is up
                            if 110 <= angle <= 140 and palm_orientation_result == "Palm Up":
                                phone_detected = True  # Phone is detected

                            # Draw hand landmarks
                            for landmark in mp_hands.HandLandmark:
                                x_hand = int(hand_landmarks.landmark[landmark.value].x * frame.shape[1])
                                y_hand = int(hand_landmarks.landmark[landmark.value].y * frame.shape[0])
                                cv2.circle(image, (x_hand, y_hand), 5, (0, 255, 0), -1)

                        elif side == "left" and hand_x < wrist[0]:  # Left hand
                            index_finger_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y]
                            thumb = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP.value].x,
                                     hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].y]

                            # Determine palm orientation using the improved method
                            palm_orientation_result = palm_orientation(wrist, thumb)

                            # Draw hand landmarks
                            for landmark in mp_hands.HandLandmark:
                                x_hand = int(hand_landmarks.landmark[landmark.value].x * frame.shape[1])
                                y_hand = int(hand_landmarks.landmark[landmark.value].y * frame.shape[0])
                                cv2.circle(image, (x_hand, y_hand), 5, (0, 255, 0), -1)

                # Print the angle and palm orientation on the image
                cv2.putText(image, f'{side.capitalize()} Arm Angle: {int(angle)} degrees', (10, 30 if side == "left" else 350), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(image, f'{side.capitalize()} Palm Orientation: {palm_orientation_result}', (10, 60 if side == "left" else 380), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Check if phone is detected and display the message
                if phone_detected:
                    cv2.putText(image, 'Phone Detected', (200, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Process left arm
            process_arm(mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                         mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                         mp_pose.PoseLandmark.RIGHT_WRIST.value, 
                         "left")

            # Process right arm
            process_arm(mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                         mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                         mp_pose.PoseLandmark.LEFT_WRIST.value, 
                         "right")

        # Display the result
        cv2.imshow('Arm and Hand Tracking', image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
