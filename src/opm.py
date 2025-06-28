import os
import sys
import traceback
import cv2
import mediapipe as mp
import numpy as np

class GarmentSizeMeasurement:
    def __init__(self, reference_image_path, known_distance_cm, movenet_model_path):
        print("Initializing GarmentSizeMeasurement...")
        try:
            # Initialize MediaPipe Pose for accurate key points
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Pose initialized successfully")
            
            # Load MoveNet model
            try:
                self.movenet = cv2.dnn.readNetFromTensorflow(movenet_model_path)
                print("MoveNet model loaded successfully")
            except Exception as e:
                print(f"Error loading MoveNet model: {e}")
                raise

            # Set up reference image and pixel-to-cm ratio
            self.reference_image_path = reference_image_path
            self.known_distance_cm = known_distance_cm
            self.pixel_to_cm_ratio = None
            self.calculate_pixel_to_cm_ratio()
            
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def calculate_pixel_to_cm_ratio(self):
        print("Calculating pixel to cm ratio...")
        ref_image = cv2.imread(self.reference_image_path)
        if ref_image is None:
            raise ValueError(f"Reference image could not be loaded from {self.reference_image_path}")
        
        gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in the reference image.")
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pixel_length = w
        self.pixel_to_cm_ratio = self.known_distance_cm / pixel_length
        print(f"Pixel to cm ratio calculated: {self.pixel_to_cm_ratio}")

    def extract_movenet_keypoints(self, frame):
        try:
            # Run MoveNet inference on the frame
            blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
            self.movenet.setInput(blob)
            output = self.movenet.forward()

            keypoints = []
            num_keypoints = output.shape[2]  # Number of keypoints

            for i in range(num_keypoints):
                x = output[0, 0, i, 0] * frame.shape[1]  # x-coordinate
                y = output[0, 0, i, 1] * frame.shape[0]  # y-coordinate
                confidence = output[0, 0, i, 2]  # Confidence score

                # Ensure confidence is a scalar
                if isinstance(confidence, np.ndarray) and confidence.size == 1:
                    confidence = confidence.item()

                # Add keypoint only if confidence is above threshold
                if confidence > 0.5:
                    keypoints.append((int(x), int(y), confidence))

            return keypoints
        except Exception as e:
            print(f"Error in extract_movenet_keypoints: {e}")
            return []

    def get_body_measurements(self, landmarks, frame_shape):
        try:
            # Shoulder width
            shoulder_width = np.linalg.norm(
                np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[12].x, landmarks[12].y])
            ) * frame_shape[1]
            shoulder_width_cm = shoulder_width * self.pixel_to_cm_ratio

            # Sleeve length
            shoulder_to_elbow = np.linalg.norm(
                np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[13].x, landmarks[13].y])
            ) * frame_shape[1]
            elbow_to_wrist = np.linalg.norm(
                np.array([landmarks[13].x, landmarks[13].y]) - np.array([landmarks[15].x, landmarks[15].y])
            ) * frame_shape[1]
            sleeve_length = shoulder_to_elbow + elbow_to_wrist
            sleeve_calibration_factor = 0.85
            sleeve_length_cm = sleeve_length * self.pixel_to_cm_ratio * sleeve_calibration_factor

            # Leg length
            hip_to_knee = np.linalg.norm(
                np.array([landmarks[23].x, landmarks[23].y]) - np.array([landmarks[25].x, landmarks[25].y])
            ) * frame_shape[1]
            knee_to_ankle = np.linalg.norm(
                np.array([landmarks[25].x, landmarks[25].y]) - np.array([landmarks[27].x, landmarks[27].y])
            ) * frame_shape[1]
            leg_length = hip_to_knee + knee_to_ankle
            leg_calibration_factor = 0.6
            leg_length_cm = leg_length * self.pixel_to_cm_ratio * leg_calibration_factor

            return {
                "shoulder_width_cm": shoulder_width_cm,
                "sleeve_length_cm": sleeve_length_cm,
                "leg_length_cm": leg_length_cm
            }
        except Exception as e:
            print(f"Error in get_body_measurements: {e}")
            return {
                "shoulder_width_cm": 0,
                "sleeve_length_cm": 0,
                "leg_length_cm": 0
            }

    def draw_measurements(self, frame, measurements):
        try:
            # Display the measurements on the frame
            cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width_cm']:.2f} cm", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Sleeve Length: {measurements['sleeve_length_cm']:.2f} cm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Leg Length: {measurements['leg_length_cm']:.2f} cm", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error in draw_measurements: {e}")

    def run_movenet_once(self, frame):
        try:
            # Perform a MoveNet forward pass
            keypoints = self.extract_movenet_keypoints(frame)

            # Initialize an empty heatmap
            heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            for keypoint in keypoints:
                x, y, confidence = keypoint
                # Scale confidence value to 0-255 for heatmap intensity
                intensity = int(confidence * 255)
                cv2.circle(heatmap, (x, y), 15, (intensity), -1)

            # Apply a colormap to the heatmap
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Overlay heatmap on original frame
            overlayed_frame = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

            return overlayed_frame
        except Exception as e:
            print(f"Error in run_movenet_once: {e}")
            return frame

    def measure_garment_size(self, video_source=0):
        print(f"Starting video capture from source: {video_source}")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Could not open video source: {video_source}")
            return

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Print frame shape for debugging
                print(f"Frame shape: {frame.shape}")

                # MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                # Generate heatmap and overlay it
                overlayed_frame = self.run_movenet_once(frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    measurements = self.get_body_measurements(landmarks, frame.shape)
                    
                    # Draw measurements on the overlayed frame
                    self.draw_measurements(overlayed_frame, measurements)

                    # Draw MediaPipe keypoints
                    for landmark in results.pose_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(overlayed_frame, (x, y), 3, (0, 255, 0), -1)

                cv2.imshow('Garment Size Measurement with Heatmaps', overlayed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in measure_garment_size: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        print("Starting application...")
        
        # File paths
        reference_image_path = "C:/Users/Vignesh/Downloads/WhatsApp Image 2024-11-18 at 10.02.40_03786f70.jpg"
        known_distance_cm = 30
        movenet_model_path = 'C:/Users/Vignesh/Downloads/graph_optt.pb'
        video_source = "C:/Users/Vignesh/Downloads/humanpose/istockphoto-638191624-640_adpp_is.mp4"
        
        # Verify files exist
        if not os.path.exists(reference_image_path):
            print(f"Reference image not found: {reference_image_path}")
            sys.exit(1)
        if not os.path.exists(movenet_model_path):
            print(f"MoveNet model not found: {movenet_model_path}")
            sys.exit(1)
        if not os.path.exists(video_source):
            print(f"Video file not found: {video_source}")
            sys.exit(1)

        print("Creating measurement tool...")
        measurement_tool = GarmentSizeMeasurement(reference_image_path, known_distance_cm, movenet_model_path)
        
        print("Starting garment measurement...")
        measurement_tool.measure_garment_size(video_source)
        
    except Exception as e:
        print(f"Main error: {e}")
        traceback.print_exc()