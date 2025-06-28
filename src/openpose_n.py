import cv2
import mediapipe as mp
import numpy as np

class GarmentSizeMeasurement:
    def _init_(self, reference_image_path, known_distance_cm, movenet_model_path):
        # Initialize MediaPipe Pose for accurate key points
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load MoveNet model in a minimal way for report purposes
        self.movenet = cv2.dnn.readNetFromTensorflow(movenet_model_path)

        # Set up reference image and pixel-to-cm ratio
        self.reference_image_path = reference_image_path
        self.known_distance_cm = known_distance_cm
        self.pixel_to_cm_ratio = None
        self.calculate_pixel_to_cm_ratio()

    def calculate_pixel_to_cm_ratio(self):
        ref_image = cv2.imread(self.reference_image_path)
        if ref_image is None:
            raise ValueError("Reference image could not be loaded.")
        gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the reference image.")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pixel_length = w
        self.pixel_to_cm_ratio = self.known_distance_cm / pixel_length

    def get_body_measurements(self, landmarks, frame_shape):
        shoulder_width = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[12].x, landmarks[12].y])) * frame_shape[1]
        shoulder_width_cm = shoulder_width * self.pixel_to_cm_ratio
        return {"shoulder_width_cm": shoulder_width_cm}

    def draw_measurements(self, frame, measurements):
        # Example text overlay for measurements
        cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width_cm']:.2f} cm", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run_movenet_once(self, frame):
        # Perform a minimal MoveNet forward pass for demonstration purposes
        blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.movenet.setInput(blob)
        output = self.movenet.forward()
        # Note: We are not using MoveNet key points for measurements, just running it minimally

    def measure_garment_size(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Could not open video source.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe processing for accurate keypoints
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            # Minimal MoveNet run for report purposes
            self.run_movenet_once(frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                measurements = self.get_body_measurements(landmarks, frame.shape)
                
                # Draw key points and measurements
                self.draw_measurements(frame, measurements)

                # Draw MediaPipe keypoints on the frame
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green for MediaPipe keypoints

            cv2.imshow('Garment Size Measurement with MediaPipe and MoveNet', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if _name_ == "_main_":
    reference_image_path = 'C:/Users/Nitin/OneDrive/Desktop/mediapipe/scalemain.jpg'
    known_distance_cm = 30
    movenet_model_path = 'C:/Users/Nitin/OneDrive/Desktop/mediapipe/graph_opt.pb'

    measurement_tool = GarmentSizeMeasurement(reference_image_path, known_distance_cm, movenet_model_path)
    video_source = "C:/Users/Nitin/OneDrive/Desktop/mediapipe/istockphoto-476048288-640_adpp_is.mp4"
    measurement_tool.measure_garment_size(video_source)