# test/test_depth.py
import os
import unittest
import numpy as np
from PIL import Image
import torch
from utility import DepthEstimator

TEST_IMG_PATH = os.path.join(
    os.path.dirname(__file__), "assets", 'image_3.jpg'
)

assert os.path.exists(TEST_IMG_PATH), (
    f"Image is not found {TEST_IMG_PATH}."
)

class TestDepthEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.estimator = DepthEstimator(fp16=False)

    def run_inference(self):
        img = Image.open(TEST_IMG_PATH)
        depth_np, depth_tensor = self.estimator.infer(img, upscale=True)
        return img, depth_np, depth_tensor

    # ---- Test case 1: Shape & dtype ------------------------------------ #
    def test_infer_output_shape_and_type(self):
        img, depth_np, depth_tensor = self.run_inference()
        
        h, w = img.size[1], img.size[0]
        
        # Kiểm tra numpy array
        self.assertIsInstance(depth_np, np.ndarray, "depth_np cần là numpy.ndarray")
        self.assertEqual(depth_np.shape, (h, w), "depth_np shape phải trùng với ảnh (H,W)")
        
        # Kiểm tra tensor
        self.assertIsInstance(depth_tensor, torch.Tensor, "depth_tensor cần là torch.Tensor")
        self.assertEqual(depth_tensor.dim(), 3, "depth_tensor cần có shape (1,H,W)")

    # ---- Test case 2: Thống kê hợp lý ---------------------------------- #
    def test_value_stats_are_ordered(self):
        _, depth_np, _ = self.run_inference()
        min_v = depth_np.min()
        mean_v = depth_np.mean()
        max_v = depth_np.max()

        self.assertLessEqual(min_v, mean_v, "Min phải ≤ Mean")
        self.assertLessEqual(mean_v, max_v, "Mean phải ≤ Max")
        self.assertGreater(max_v - min_v, 0, "Khoảng cách giá trị phải > 0")

    # ---- Test case 3: Hàm heatmap trả về đúng định dạng ---------------- #
    def test_get_heatmap_returns_color_image(self):
        img, depth_np, _ = self.run_inference()
        heatmap_bgr, stats = self.estimator.get_heatmap(depth_np)

        h, w = img.size[1], img.size[0]
        self.assertEqual(heatmap_bgr.shape, (h, w, 3), "Heatmap phải có shape (H, W, 3)")
        self.assertTrue(all(isinstance(x, float) for x in stats), "Stats phải là tuple float")

# ---- Phần hiển thị hình ảnh (chạy độc lập) -------------------------- #
def visualize_results(show_compare: bool = True):
    """
    Hiển thị:
      • Depth map (gray)
      • Heat‑map hiển thị sai màu (BGR đưa thẳng vào matplotlib)
      • Heat‑map hiển thị đúng màu (RGB)

    Parameters
    ----------
    show_compare : bool
        True  → vẽ 3 khung (BGR vs RGB) để so sánh
        False → chỉ vẽ depth + heat‑map RGB
    """
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    import os

    estimator = DepthEstimator(fp16=False)

    img = Image.open(TEST_IMG_PATH)
    depth_np, _ = estimator.infer(img, upscale=True)

    # lấy heat‑map BGR và RGB từ hàm mới
    heatmap_bgr, stats = estimator.get_heatmap(depth_np, rgb=False)
    heatmap_rgb, _    = estimator.get_heatmap(depth_np, rgb=True)

    # -------- Vẽ -------------------------------------------------------
    n_cols = 3 if show_compare else 2
    plt.figure(figsize=(5 * n_cols, 4))

    # Depth map (gray)
    plt.subplot(1, n_cols, 1)
    plt.title("Depth Map")
    plt.imshow(depth_np, cmap="gray")
    plt.axis("off")

    if show_compare:
        # Heat‑map BGR đưa thẳng -> màu sai
        plt.subplot(1, n_cols, 2)
        plt.title("Heat‑map (BGR ‑ màu lệch)")
        plt.imshow(heatmap_bgr)          # matplotlib nghĩ là RGB
        plt.axis("off")

        # Heat‑map RGB -> màu đúng
        plt.subplot(1, n_cols, 3)
        plt.title("Heat‑map (RGB – chuẩn màu)")
        plt.imshow(heatmap_rgb)
        plt.axis("off")
    else:
        # chỉ vẽ heat‑map đúng màu
        plt.subplot(1, n_cols, 2)
        plt.title("Heat‑map (RGB)")
        plt.imshow(heatmap_rgb)
        plt.axis("off")

    print(
        f"Stats: min={stats[0]:.2f} m   max={stats[1]:.2f} m   mean={stats[2]:.2f} m"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys

    # Nếu gọi kèm --show thì chỉ hiển thị ảnh, bỏ qua test
    if "--show" in sys.argv:
        visualize_results()
    else:
        # Chạy unit test
        unittest.main()
