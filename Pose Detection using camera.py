import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose and Drawing Utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose_live():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while True:  # Use an infinite loop
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(frame_rgb)
            
            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )
            
            # Display the frame
            cv2.imshow('Pose Detection', frame)
            
            # Quit when 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):  
                print("Exiting pose detection...")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera and windows released properly.")

# Call the live detection function
detect_pose_live()
