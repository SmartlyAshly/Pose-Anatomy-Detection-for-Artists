import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect pose
    results = pose.process(image_rgb)
    
    # Check if pose landmarks are detected
    if results.pose_landmarks:
        print("Pose landmarks detected.")
        # Draw pose landmarks on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # Display the annotated image
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No pose landmarks detected.")

# Example usage: Provide the path to an image
image_path = "SS.jpg"  # Replace with your image path
detect_pose(image_path)
