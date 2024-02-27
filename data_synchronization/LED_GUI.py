import numpy as np
from skimage import io, filters, measure, color, morphology
import cv2
import matplotlib.pyplot as plt

class ImageCropper:
    def __init__(self, image_array):
        """
        Parameters:
        - image_array (ndarray): A numpy array representing the image to be cropped.
                                 This array is expected to be in the format readable by OpenCV,
                                 typically (height, width, channels).
        """
        self.image = image_array.copy()
        self.clone = image_array.copy()
        self.ref_point = []  # To store rectangle coordinates
        self.cropping = False  # Flag to indicate that cropping is being performed

    def click_and_crop(self, event, x, y, flags, param):
        """
        The callback function to handle mouse events for selecting the cropping rectangle.

        This method updates the `ref_point` list with the coordinates of the rectangle
        drawn by the user's mouse actions. It also updates the `cropping` flag and the
        displayed image to show the selected area.

        Parameters:
        - event: The type of mouse event (e.g., left button down, left button up).
        - x (int): The x-coordinate of the mouse event.
        - y (int): The y-coordinate of the mouse event.
        - flags: Any relevant flags passed by OpenCV. This parameter is not used in this method.
        - param: Additional parameters passed by OpenCV. This parameter is not used in this method.
        """
        # If the left mouse button was clicked, record the starting (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True

        # Check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # Record the ending (x, y) coordinates and indicate that the cropping operation is finished
            self.ref_point.append((x, y))
            self.cropping = False

            # Draw a rectangle around the region of interest
            cv2.rectangle(self.image, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            cv2.imshow("image", self.image)

    def show_and_crop_image(self):
        """
        Display the image and allow the user to select a region to crop. The method captures
        mouse events to define the corners of the rectangle. The user can reset the selection
        by pressing 'r' and confirm the selection by pressing 'c'.

        Returns:
        tuple: A tuple of (cropped_image, list of reference points defining the selected rectangle).
               If no selection is made, returns None.
        """
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)

        # Keep looping until the 'q' key is pressed
        while True:
            # Display the image and wait for a keypress
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            # If the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                self.image = self.clone.copy()

            # If the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        cv2.destroyAllWindows()

        # If there are two reference points, then crop the region of interest from the image
        if len(self.ref_point) == 2:
            roi = self.clone[self.ref_point[0][1]:self.ref_point[1][1], self.ref_point[0][0]:self.ref_point[1][0]]
            return roi, self.ref_point

if __name__ == "__main__":
    # Example usage
    image_path = '/home/mpicek/repos/master_project/test_data/camera/C0170T01.JPG'
    image_array = cv2.imread(image_path)  # Ensure you read the image with OpenCV
    cropper = ImageCropper(image_array)
    roi, ref_point = cropper.show_and_crop_image()

    if roi is not None:
        # Display or process the ROI
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
