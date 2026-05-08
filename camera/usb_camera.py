import cv2
import numpy as np
from .camera import Camera

class UsbCamera(Camera):

    def _find_best_contour(self,contours):
        """
        Finds the contour that matches set
        aspect ratios best that has 4 corners.
        """

        # Const variables
        ASPECT_HORIZONTAL = 5/7
        ASPECT_VERTICAL = 7/5
        ASPECT_TOLERANCE = .2

        best_area = 0           # current best area (helps find the best contour)
        best_contour = None     # results

        for i, contour in enumerate(contours):

            perimeter = cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Checks if the countour has 4 sides
            if len(approx) !=4:
                continue

            # Checks to see if the contour is the main one 
            # taking up most of the picture
            area = cv2.contourArea(approx)
            if area < 20000:
                continue

            # Checks if the aspect ratio of the card
            x, y, width, height = cv2.boundingRect(approx)
            if height > 0:
                aspect = width / height
            else:
                aspect = 0

            if not (ASPECT_HORIZONTAL - ASPECT_TOLERANCE < aspect < ASPECT_HORIZONTAL + ASPECT_TOLERANCE or
                ASPECT_VERTICAL - ASPECT_TOLERANCE < aspect < ASPECT_VERTICAL + ASPECT_TOLERANCE):
                continue 

            # Assumes the countour with the biggest area is
            # the card
            if best_area < area:
                best_area = area
                best_contour = approx
            
        return best_contour

    def connect_carmera(self, camera_number):
        """
        Implements Camer.connect_camera(self,camera_number)
        """

        # Check the type of camera_number
        if not isinstance(camera_number, int):
            raise TypeError("Inputted camera number must be an integer.")

        # Checks if the camera connected successfully
        camera = cv2.VideoCapture(camera_number)
        if not camera.isOpened():
            raise Exception(f"Failed to open camera {camera_number}")
        
        return camera

    def capture_image(self, camera):
        """
        Implements Camera.caputre_image(self,camera)
        """

        if not isinstance(camera,cv2.VideoCapture):
            raise TypeError("Inputted camera is not of type VideoCapture.")

        # Checks if the image was captured successfully
        ret, image = camera.read()
        if ret == False:
            raise Exception("Failed to capture image from the connected camera.")
        
        return image
    
    def prepare_image(self, image):
        """
        Implements Camera.prepared_image(self, image)
        """

        # Checks the type of the image
        if not isinstance(image, np.ndarray):
            raise TypeError("Inputted image must be a NumPyArray.")
        
        # Checks if the image needs to be changed to gray
        if image.ndim == 3:
            if image.shape[2] != 3:
                raise ValueError("Color image must have 3 channels.")
            
            grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            grey = image.copy()

        blur = cv2.GaussianBlur(grey,(5,5),0)
        canny = cv2.Canny(blur,50,150)                          # Thresholds might be to high

        
        # Connect edges
        kernel = np.ones((3, 3), np.uint8)                      # Increasing the matrix size increases thickness
        dilated = cv2.dilate(canny, kernel, iterations=2)       # Might chanage back to 1 iteration if the edges connect to much

        return dilated
    
    def find_card(self, image):
        """
        Implements Camera.prepared_image(self, image)
        """
        # Checks the type of the image
        if not isinstance(image, np.ndarray):
            raise TypeError("Inputted image must be a NumPyArray.")
        
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        result_contour = self._find_best_contour(contours)

        return result_contour

    def order_points(self, points):
        """
        Implements Camera.order_points(self, points)
        """

        # Checks if points are a numpy array
        if not isinstance(points,np.ndarray):
            raise TypeError("Inputted points must be a NumPyArray.")

        # Checks if the input needs to be resized
        if points.shape == (4,1,2):
            points = points.reshape(4,2)
        # Checks if the input is another valid shape
        elif points.shape == (4,2):
            points = points.copy()
        # If the points array is not valid for either than it throws an error
        else:
            raise Exception(f"Invalid points shape {points.shape}.")
        
        rect = np.zeros((4, 2), dtype="float32")

        sum = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        rect[0] = points[np.argmin(sum)]  # top-left
        rect[1] = points[np.argmin(diff)]  # top-right
        rect[2] = points[np.argmax(sum)]  # bottom-right
        rect[3] = points[np.argmax(diff)]  # bottom-left

        return rect

    def get_card(self, image, points):
        """
        Implements Camera.get_card(self, points)
        """

        # Checks if points are a numpy array
        if not isinstance(points,np.ndarray):
            raise TypeError("Inputted points must be a NumPyArray.")
        
        # verify the points are ordered
        rect = self.order_points(points)

        # top left -> bottom-left
        (tl,tr,br,bl) = rect

        # Find the widest part of the card
        widthA = np.linalg.norm(tr - tl)
        widthB = np.linalg.norm(br - bl)
        maxWidth = int(max(widthA, widthB))

        # Find the tallest part of the card
        heightA = np.linalg.norm(tl - bl)
        heightB = np.linalg.norm(tr - br)
        maxHeight = int(max(heightA, heightB))

        # Convert the width/height to an array
        # To define the shape of the card
        dst = np.array([
            [0,0],
            [maxWidth - 1,0],
            [maxWidth - 1,maxHeight - 1],
            [0,maxHeight - 1]
        ], dtype=np.float32)

        # Computes the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(rect,dst)

        # Perform the transformation
        card = cv2.warpPerspective(image,matrix, (maxWidth, maxHeight))

        return card

    
