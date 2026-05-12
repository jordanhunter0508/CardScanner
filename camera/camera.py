"""
This interface can be used to capture an image
and process the image to display a single card
"""
import numpy as np
import cv2
from abc import ABC, abstractmethod

class Camera(ABC):

    @abstractmethod
    def connect_camera(self, camera_number: int) -> cv2.VideoCapture:
        """
        Establishes a camera connection

        Args:
            camera_number: The number of the camera the user wants to use.
        
        Returns:
            A cv2.VideoCapture object
        """

    @abstractmethod
    def capture_image(self, camera: cv2.VideoCapture) -> cv2.typing.MatLike:
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
    def prepare_image(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
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
    def find_card(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike | None:
        """
        Finds the edges of the cards with canny

        Args:
            image: The captured image.

        Returns:
            Coordinates of the detected card as a tuple.
        """
        pass

    @abstractmethod
    def order_points(self, points: np.ndarray) -> np.ndarray:
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
    def get_card(self, image: cv2.typing.MatLike, points: np.ndarray) -> cv2.typing.MatLike:
        """
        Uses the coordinates to find and return
        the image of a card
        
        Args: 
            image: original image taken from capture_image
            points: NumPy array of where the card is located.

        Returns:
            Image of a card.
        """
        pass