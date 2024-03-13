
import numpy as np
import face_recognition
import cv2

def get_face_coordinate_system(detection_result, color_image):
    """
    Establishes a face coordinate system based on detected facial landmarks.

    Parameters:
    - detection_result: The result object from a face detection model, expected to have a 'face_landmarks' attribute.
    - color_image: The resized color image numpy array on which detection was performed, shape (H, W, C).

    Returns:
    - tuple: Contains the origin of the face coordinate system in camera coordinates (numpy array),
             the transformation matrix from face to camera coordinates (numpy array),
             and the transformation matrix from camera to face coordinates (numpy array).
             The face coordinate system is defined with orthogonal bases where:
             - The X-axis (eye_vector_orthogonalized) is normalized and orthogonalized from right to left eye direction.
             - The Y-axis (nose_vector) is normalized, representing the direction from the nose base to the top of the nose.
             - The Z-axis is orthogonal to both X and Y axes, completing the right-handed coordinate system.
             The origin is set at a specific facial landmark with adjusted Y-axis direction to account for image coordinate conventions.
    """

    # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
    face_coordinates_origin = np.array([
        detection_result.face_landmarks[0][1].x * color_image.shape[1],
        -detection_result.face_landmarks[0][1].y * color_image.shape[0],
        detection_result.face_landmarks[0][1].z * color_image.shape[1],
    ])

    nose_vector = np.zeros((3,))

    nose_vector[0] = (detection_result.face_landmarks[0][8].x - detection_result.face_landmarks[0][2].x)
    # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
    nose_vector[1] = -(detection_result.face_landmarks[0][8].y - detection_result.face_landmarks[0][2].y)
    nose_vector[2] = (detection_result.face_landmarks[0][8].z - detection_result.face_landmarks[0][2].z)
    nose_vector = nose_vector / np.linalg.norm(nose_vector)
    
    # eye vector going from right patient's eye to the left patient's eye (for us the vector goes to the right
    # when the patient is looking at us)
    eye_vector = np.zeros((3,))
    eye_vector[0] = (detection_result.face_landmarks[0][263].x - detection_result.face_landmarks[0][33].x)
    # mediapipe returns y going down, we want it to go up (thus the minus sign in y coordinate)
    eye_vector[1] = -(detection_result.face_landmarks[0][263].y - detection_result.face_landmarks[0][33].y)
    eye_vector[2] = (detection_result.face_landmarks[0][263].z - detection_result.face_landmarks[0][33].z)
    eye_vector = eye_vector / np.linalg.norm(eye_vector)

    # we want to find the orthogonalized eye vector - Let's use Gram Schmidt process
    # so we subtract the projection of the eye vector to the nose vector from the eye vector
    # projection is <v1 . v2> * v1 (it's how much it's projeceted times v1 which is the direction
    # of the projection). This will be subtracted from the original vector to get the orthogonalized:
    # v2_orth = v2 - <v1 . v2> * v1
    # (and we suppose that both vectors are normalized, so we don't have to divide by the norms)
    eye_vector_orthogonalized = eye_vector - np.dot(eye_vector, nose_vector) * nose_vector
    # and normalize it to be sure
    eye_vector_orthogonalized = eye_vector_orthogonalized / np.linalg.norm(eye_vector_orthogonalized)

    third_orthogonal_vector = np.cross(nose_vector, eye_vector_orthogonalized)
    third_orthogonal_vector = third_orthogonal_vector / np.linalg.norm(third_orthogonal_vector)

    # this matrix transforms points in face coordinates to the camera coordinates
    face2cam = np.column_stack((eye_vector_orthogonalized, nose_vector, third_orthogonal_vector))

    # this matrix (the inverse) transforms points in camera coordinates to the face coordinates
    # it has an inverse for as it was a orthonormal matrix (it's actually its transpose..)
    cam2face = np.linalg.inv(face2cam)

    return face_coordinates_origin, face2cam, cam2face

def get_forehead_point(detection_result, color_image):
    """
    Calculate the normalized 3D point of the forehead from detection results.

    Parameters:
    - detection_result: The result object from a face detection model, expected to have a 'face_landmarks' attribute.
    - color_image: The resized color image numpy array on which detection was performed, shape (H, W, C).

    Returns:
    - np.array: A numpy array containing the normalized (x, y, z) coordinates of the forehead point in the image space. 
                
    - Detect the point on forehead (and z is relative, but we don't care about it because a person
      moves the head in an "angular" way and not back and forth. And the point is on the forehead,
      so the angle of the head is going to be very similar to the z coordinate. It's an approximation
      that should be good enough for our purposes)
    - We need to normalize the coordinates because mediapipe returns just the relative coordinates to the
      size of the image. So we multiply it by the width and height of the image (cropped area only in our case
      as mediapipe operates only on the cropped image!!). And, according to mediapipe docs, the z coordinate
      is approximately of the same magnitude as x
    """

    return np.array([
        detection_result.face_landmarks[0][8].x * color_image.shape[1],
        detection_result.face_landmarks[0][8].y * color_image.shape[0],
        detection_result.face_landmarks[0][8].z * color_image.shape[1],
    ])

class NoFaceDetectedException(Exception):
    pass

def recognize_patient(color_image, patients_encoding):
    """
    Parameters:
    - color_image (numpy.ndarray): The color image (RGB) in which to recognize the patient's face. Expected shape is (height, width, 3).
    - patients_encoding (np.ndarray): Encoding of the patient's face(s)

    Returns:
    - tuple: A tuple of 4 integers representing the location of the patient's face in the image: (top, right, bottom, left).

    Raises:
    - NoFaceDetectedException: If no faces are detected in the provided image.

    Note:
    - The face recognition model used is 'cnn' - a slower but much better model for face localization than the default 'hog' model.
    """

    # face_recognition library works with BGR images
    locations = face_recognition.face_locations(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), model='cnn')
    encodings = face_recognition.face_encodings(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), known_face_locations=locations)

    if len(locations) == 0:
        raise NoFaceDetectedException("No face detected in the image")
    
    face_distances = face_recognition.face_distance(encodings, patients_encoding)
    patients_face_index = np.argmin(face_distances)
    patients_face_location = locations[patients_face_index]

    return patients_face_location