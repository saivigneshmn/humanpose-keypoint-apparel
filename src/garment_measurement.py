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
        
        # Load MoveNet model
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

    def extract_movenet_keypoints(self, frame):
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
                confidence = confidence.item()  # Convert to scalar if it's an array

            # Add keypoint only if confidence is above threshold
            if confidence > 0.5:
                keypoints.append((int(x), int(y), confidence))

        return keypoints

    def get_body_measurements(self, landmarks, frame_shape):
        # Shoulder width MNSV
        shoulder_width = np.linalg.norm(
            np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[12].x, landmarks[12].y])
        ) * frame_shape[1]
        shoulder_width_cm = shoulder_width * self.pixel_to_cm_ratio

        # Sleeve length as the sum of shoulder to elbow and elbow to wrist
        shoulder_to_elbow = np.linalg.norm(
            np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[13].x, landmarks[13].y])
        ) * frame_shape[1]
        elbow_to_wrist = np.linalg.norm(
            np.array([landmarks[13].x, landmarks[13].y]) - np.array([landmarks[15].x, landmarks[15].y])
        ) * frame_shape[1]
        sleeve_length = shoulder_to_elbow + elbow_to_wrist
        sleeve_calibration_factor = 0.85  # Calibration for sleeve length precision
        sleeve_length_cm = sleeve_length * self.pixel_to_cm_ratio * sleeve_calibration_factor

        # Leg length as the sum of hip to knee and knee to ankle
        hip_to_knee = np.linalg.norm(
            np.array([landmarks[23].x, landmarks[23].y]) - np.array([landmarks[25].x, landmarks[25].y])
        ) * frame_shape[1]
        knee_to_ankle = np.linalg.norm(
            np.array([landmarks[25].x, landmarks[25].y]) - np.array([landmarks[27].x, landmarks[27].y])
        ) * frame_shape[1]
        leg_length = hip_to_knee + knee_to_ankle
        leg_calibration_factor = 0.6  # Calibration for leg length precision
        leg_length_cm = leg_length * self.pixel_to_cm_ratio * leg_calibration_factor

        return {
            "shoulder_width_cm": shoulder_width_cm,
            "sleeve_length_cm": sleeve_length_cm,
            "leg_length_cm": leg_length_cm
        }

    def draw_measurements(self, frame, measurements):
        # Display the measurements on the frame
        cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width_cm']:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sleeve Length: {measurements['sleeve_length_cm']:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Leg Length: {measurements['leg_length_cm']:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run_movenet_once(self, frame):
        # Perform a minimal MoveNet forward pass for demonstration purposes
        keypoints = self.extract_movenet_keypoints(frame)
        return keypoints

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
            keypoints = self.run_movenet_once(frame)

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

            # Optionally, display MoveNet keypoints on the frame
            for keypoint in keypoints:
                x, y, confidence = keypoint
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red for MoveNet keypoints

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
    video_source = "C:/Users/Nitin/OneDrive/Desktop/mediapipe/istockphoto-465625162-640_adpp_is.mp4"
    measurement_tool.measure_garment_size(video_source)
