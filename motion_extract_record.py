import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from typing import List, Tuple

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def project_to_frontal(point: np.ndarray) -> np.ndarray:
    """
    Project a MediaPipe 3D landmark onto the camera's frontal image plane.
    """
    point = np.array(point)
    return np.array([point[0], point[1], 0.0])


# Function to calculate angle between three points
def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the frontal-plane angle between three MediaPipe points.
    The angle is calculated at point b.
    """
    a = project_to_frontal(a)
    b = project_to_frontal(b)
    c = project_to_frontal(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0 or np.isnan(denominator):
        return np.nan

    # Calculate cosine of angle using dot product
    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

# Function to save pose landmarks to CSV
def save_to_csv(landmarks_history: List[List[Tuple[float, float, float]]], filename: str = "pose_data.csv"):
    """
    Save pose landmarks history to a CSV file.
    Each row represents a frame, and each column represents x, y, z coordinates of a landmark.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = []
        for i in range(33):  # MediaPipe Pose has 33 landmarks
            header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
        writer.writerow(header)
        
        # Write data
        for landmarks in landmarks_history:
            row = []
            for landmark in landmarks:
                row.extend(landmark)
            writer.writerow(row)
    
    print(f"Data saved to {filename}")

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set video resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize MediaPipe Pose with higher detection and tracking confidence
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        landmarks_history = []
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Convert the BGR image to RGB and process it with MediaPipe Pose
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert the image back to BGR and make it writeable
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Get frame dimensions
            h, w, c = image.shape
            
            if results.pose_landmarks:
                # Extract landmarks for current frame
                frame_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel values for visualization
                    px, py = int(landmark.x * w), int(landmark.y * h)
                    # Store the raw (x, y, z) coordinates
                    frame_landmarks.append((landmark.x, landmark.y, landmark.z))
                
                landmarks_history.append(frame_landmarks)
                
                # Draw pose landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Calculate knee angles (both left and right)
                # Left knee: hip (23), knee (25), ankle (27)
                left_hip = np.array([
                    results.pose_landmarks.landmark[23].x,
                    results.pose_landmarks.landmark[23].y,
                    results.pose_landmarks.landmark[23].z
                ])
                left_knee = np.array([
                    results.pose_landmarks.landmark[25].x,
                    results.pose_landmarks.landmark[25].y,
                    results.pose_landmarks.landmark[25].z
                ])
                left_ankle = np.array([
                    results.pose_landmarks.landmark[27].x,
                    results.pose_landmarks.landmark[27].y,
                    results.pose_landmarks.landmark[27].z
                ])
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                
                # Right knee: hip (24), knee (26), ankle (28)
                right_hip = np.array([
                    results.pose_landmarks.landmark[24].x,
                    results.pose_landmarks.landmark[24].y,
                    results.pose_landmarks.landmark[24].z
                ])
                right_knee = np.array([
                    results.pose_landmarks.landmark[26].x,
                    results.pose_landmarks.landmark[26].y,
                    results.pose_landmarks.landmark[26].z
                ])
                right_ankle = np.array([
                    results.pose_landmarks.landmark[28].x,
                    results.pose_landmarks.landmark[28].y,
                    results.pose_landmarks.landmark[28].z
                ])
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Display knee angles on the image
                cv2.putText(image, f"Left Knee Angle: {left_knee_angle:.1f}°", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Right Knee Angle: {right_knee_angle:.1f}°", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(image, "Press 'q' to quit and save data", 
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow('MediaPipe Pose Estimation', image)
            
            # Exit on 'q' press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save landmarks history to CSV if we have data
        if landmarks_history:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_to_csv(landmarks_history, f"pose_data_{timestamp}.csv")

if __name__ == "__main__":
    main()
