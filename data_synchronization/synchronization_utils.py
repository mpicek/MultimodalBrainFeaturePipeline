
import numpy as np
import face_recognition
import cv2
from scipy.signal import resample
import plotly.graph_objects as go

log_table_columns = [
        # the following are for the extraction
        'path_bag',
        'output_path',
        'sync_strategy', # acc, 
        'failed',
        'date',
        'avg_quality',
        'first_frame_face_detected',
        'face_location',
        'face_patient_distance',
        'Exception5000',
        'unknown_error_extraction',
        'relevant_wisci_files',
        # and these are for the synchronization
        'path_wisci',
        'normalized_corr',
        'lag',
        'fps_bag',
        'unknown_error_synchronization',
        'peaks_per_million',
        'best_second_largest_corr_peak',
        'second_best_wisci_corr',
        'second_best_wisci_peaks_per_million',
        'second_best_wisci_path',
        'synchronization_failed',
    ]

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

    return patients_face_location, np.argmin(face_distances)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def rotation_x_axis_counterclws(angle):
    """
    angle in degrees
    """
    deg_rad = 2*np.pi*angle / 360
    return np.array([
            [1, 0, 0],
            [0, np.cos(deg_rad), -np.sin(deg_rad)],
            [0, np.sin(deg_rad), np.cos(deg_rad)]
        ])

def rotation_y_axis_counterclws(angle):
    """
    angle in degrees
    """
    deg_rad = 2*np.pi*angle / 360
    return np.array([
            [np.cos(deg_rad), 0, np.sin(deg_rad)],
            [0, 1, 0],
            [-np.sin(deg_rad), 0, np.cos(deg_rad)]
        ])

def rotation_z_axis_counterclws(angle):
    """
    angle in degrees
    """
    deg_rad = 2*np.pi*angle / 360
    return np.array([
            [np.cos(deg_rad), -np.sin(deg_rad), 0],
            [np.sin(deg_rad), np.cos(deg_rad), 0],
            [0, 0, 1]
        ])

def preprocess_accelerometer_data(data, box_size):

    x_acc = smooth(data[:, 0], box_size)
    y_acc = smooth(data[:, 1], box_size)
    z_acc = smooth(data[:, 2], box_size)
    x_acc = np.gradient(x_acc)
    y_acc = np.gradient(y_acc)
    z_acc = np.gradient(z_acc)

    x_acc = smooth(x_acc, box_size)
    y_acc = smooth(y_acc, box_size)
    z_acc = smooth(z_acc, box_size)

    return np.column_stack([x_acc, y_acc, z_acc])

def preprocess_video_signals(forehead_points, quality, cam2face, original_time, desired_realsense_fps, box_size):

    face2cam = np.linalg.inv(cam2face)

    # we suppose that the Wisci is just 45deg rotated in the y direction 
    # R_y = rotation_y_axis_counterclws(45)
    R_x = rotation_x_axis_counterclws(10)
    R_z = rotation_z_axis_counterclws(10)
    R_y = rotation_y_axis_counterclws(-33.29)

    in_face_space = np.dot(face2cam, forehead_points.T).T
    rotated_by_y_cam = np.dot(R_z, in_face_space.T).T
    rotated_by_z_cam = np.dot(R_y, rotated_by_y_cam.T).T
    in_wisci_space = np.dot(R_x, rotated_by_z_cam.T).T

    # Assuming `original_signal` is your NumPy array containing the signal
    original_signal = in_wisci_space  # Your signal data
    original_rate = in_wisci_space.shape[0] / original_time  # The original sampling rate

    # Calculate the duration of your signal
    num_samples = original_signal.shape[0]
    duration = num_samples / original_rate

    # Calculate the new number of samples required for the desired rate
    new_num_samples = int(np.round(duration * desired_realsense_fps))

    # Use scipy's resample to adjust the signal
    resampled_signal = resample(original_signal, new_num_samples)

    resampled_quality = resample(quality, new_num_samples)

    x_smoothed = smooth(resampled_signal[:, 0], box_size)
    y_smoothed = smooth(resampled_signal[:, 1], box_size)
    z_smoothed = smooth(resampled_signal[:, 2], box_size)

    x_gradient = np.gradient(x_smoothed)
    y_gradient = np.gradient(y_smoothed)
    z_gradient = np.gradient(z_smoothed)

    x_gradient_smoothed = smooth(x_gradient, box_size)
    y_gradient_smoothed = smooth(y_gradient, box_size)
    z_gradient_smoothed = smooth(z_gradient, box_size)

    x_gradient_smoothed = np.gradient(x_gradient_smoothed)
    y_gradient_smoothed = np.gradient(y_gradient_smoothed)
    z_gradient_smoothed = np.gradient(z_gradient_smoothed)

    x_gradient_smoothed = smooth(x_gradient_smoothed, box_size)
    y_gradient_smoothed = smooth(y_gradient_smoothed, box_size)
    z_gradient_smoothed = smooth(z_gradient_smoothed, box_size)

    x_gradient_smoothed = np.gradient(x_gradient_smoothed)
    y_gradient_smoothed = np.gradient(y_gradient_smoothed)
    z_gradient_smoothed = np.gradient(z_gradient_smoothed)

    return np.column_stack([x_gradient_smoothed, y_gradient_smoothed, z_gradient_smoothed]), resampled_quality

def find_data_in_wisci(mat, looked_for, looked_for2=None):
    """
    Find data in the given mat file based on the provided criteria.

    Parameters:
    mat (dict): The mat file containing the data.
    looked_for (str): The string to look for in the channel names.
    looked_for2 (str, optional): An additional string to look for in the channel names.

    Returns:
    tuple: A tuple containing the stream number and index of the found data. If not found, raises a ValueError.

    Raises:
    ValueError: If the data specified by `looked_for` is not found in the mat file.
    """

    # Iterate over the stream numbers
    for stream_nb in mat['STREAMS'].dtype.names: 
        print(stream_nb)

        # Iterate over the info types
        for info_type in mat['STREAMS'][stream_nb][0][0].dtype.names:
            print(f"\t {info_type}")
            
            # Check if the info type is 'ch_names'
            if info_type == 'ch_names':
                print(mat['STREAMS'][stream_nb][0][0][info_type][0][0])
                
                # Iterate over the channel names
                for i, ch_name in enumerate(mat['STREAMS'][stream_nb][0][0][info_type][0][0]):
                    print(ch_name)
                    
                    # Check if the looked_for or looked_for2 strings are present in the channel name
                    if looked_for.lower() in ch_name.lower() or (looked_for2 and looked_for2.lower() in ch_name.lower()):
                        print(f"Found {looked_for} in {ch_name}")
                        return stream_nb, i

    raise ValueError(f"Data {looked_for} not found in the mat file.")

