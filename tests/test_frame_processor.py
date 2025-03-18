import unittest
import cv2
import numpy as np
from utils.frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = FrameProcessor()
        self.test_frame = cv2.imread('sample_image.jpg')  # Thay thế bằng một hình ảnh mẫu

    def test_adjust_contrast(self):
        adjusted_frame = self.processor.adjust_contrast(self.test_frame)
        self.assertIsNotNone(adjusted_frame)

    def test_reduce_noise(self):
        denoised_frame = self.processor.reduce_noise(self.test_frame)
        self.assertIsNotNone(denoised_frame)

    def test_preprocess_frame(self):
        processed_frame = self.processor.preprocess_frame(self.test_frame)
        self.assertIsNotNone(processed_frame)

if __name__ == "__main__":
    unittest.main()