"""
This interface can be used to capture an image
and process the image to display a single card
"""
import numpy as np

from abc import ABC, abstractmethod

class Camera(ABC):

    @abstractmethod
    def connect_carmera(self, camera_number):
        """
        Establishes a camera connection

        Args:
            carmera_number: The number of the camera the user wants to use.
        
        Returns:
            A cv2.VideoCapture object
        """

    @abstractmethod
    def capture_image(self, camera):
        """
        Take a picture using the connected camera
        Keeps the standard size

        Args:
            camera: The number of the camera the user wants to use.

        Returns:
            A raw image taken from the inputted camera
        """
        pass

    @abstractmethod
    def prepare_image(self, image):
        """
        Converts the image grayscale and blurs the image
        to get ready for processing

        Args:
            image: The captured image.

        Returns:
            A binary image of the inputted image where the edges are highlighted
        """
        pass
    
    @abstractmethod
    def find_card(self, image):
        """
        Finds the edges of the cards with canny

        Args:
            image: The captured image.

        Returns:
            Coordinates of the detected card as a tuple.
        """
        pass

    @abstractmethod
    def order_points(self, points):
        """
        Orders the points from top-left, top-right,
        bottom-right, bottom-left
        
        Args: 
            points: contour that will be ordered

        Returns:
            An array of ordered points
        """
        pass

    @abstractmethod
    def get_card(self, image, points):
        """
        Uses the coordinates to find and return
        the image of a card
        
        Args: 
            image: orignial image taken from capture_image-
            points: NumPy array of where the card is located.

        Returns:
            Image of a card.
        """
        pass