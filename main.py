import cv2
import dlib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import face_recognition


predictor_path = "inference/pre-trainedmodels/shape_predictor_68_face_landmarks.dat"


# Step 2: Face Detection
def detect_faces(image_path):
    image = cv2.imread(image_path)
    boxes = face_recognition.face_locations(image, model='cnn')
    encodings = face_recognition.face_encodings(image, boxes)
    return boxes, encodings


# Step 3: Face Alignment (You may use a dedicated alignment method)
# Step 3: Face Alignment (Using dlib)
def align_face(face_image, face_landmarks):
    # Initialize the face aligner from dlib
    face_aligner = dlib.face_alignment.FaceAligner(dlib.get_frontal_face_detector(), predictor_path)

    # Align the face based on detected landmarks
    aligned_face = face_aligner.align(face_image, face_image, face_landmarks)

    return aligned_face


# Step 4: Feature Extraction
def extract_features(image, boxes):
    # Preprocess the face image (e.g., resize, normalize pixel values)
    processed_face = preprocess_image(face)

    # Load your pre-trained deep learning model (e.g., FaceNet, VGGFace, ArcFace)
    # Replace 'load_pretrained_model' with the actual code to load your model.
    model = FaceNet()

    # Extract features from the face using the model
    features = model.predict(np.expand_dims(boxes, axis=0))

    # Ensure the features have the desired dimensionality (e.g., 128-dimensional vectors)
    # Modify this part based on the output dimension of your model.
    if features.shape[1] != 128:
        raise ValueError("The model should output 128-dimensional feature vectors.")

    return features[0]


# Step 5: Clustering Algorithm
def cluster_faces(features):
    # Apply DBSCAN for clustering
    # Normalize the feature vectors (important for DBSCAN)
    normalized_features = StandardScaler().fit_transform(features)

    # Apply DBSCAN
    eps = 0.5  # The maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples = 5  # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(normalized_features)

    # Get cluster labels (e.g., -1 for noise points, 0, 1, 2, ... for clusters)
    labels = db.labels_

    return labels


