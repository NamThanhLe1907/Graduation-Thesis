from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, scale_factor=0.325):
        """
        Khởi tạo DepthEstimator với scale_factor được tính từ hiệu chuẩn.
        scale_factor: hệ số nhân hiệu chuẩn để chuyển đổi giá trị depth sang đơn vị mét
        """
        self.scale_factor = scale_factor
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", use_fast=True)

    def infer(self, image):
        """
        Thực hiện suy luận độ sâu trên một ảnh đầu vào (PIL Image).
        Trả về:
          - calibrated_depth: mảng numpy sau khi hiệu chuẩn
          - output: kết quả đầy đủ của pipeline inference
        """
        output = self.pipe(image)
        # Lấy tensor predicted_depth
        pred_depth = output["predicted_depth"]
        # Chuyển đổi tensor sang numpy array nếu cần
        try:
            if hasattr(pred_depth, "cpu"):
                pred_depth_np = pred_depth.cpu().numpy()
            else:
                pred_depth_np = np.array(pred_depth)
        except Exception as e:
            pred_depth_np = np.array(pred_depth)
        # Hiệu chuẩn độ sâu sử dụng scale_factor
        calibrated_depth = pred_depth_np * self.scale_factor
        return calibrated_depth, output

    def get_heatmap(self, calibrated_depth):
        """
        Tạo heatmap màu từ mảng độ sâu đã hiệu chuẩn.
        Trả về:
         - heatmap: ảnh heatmap dạng màu (BGR)
         - stats: tuple chứa (min_val, max_val, mean_val)
        """
        min_val = calibrated_depth.min()
        max_val = calibrated_depth.max()
        mean_val = calibrated_depth.mean()

        # Chuẩn hóa độ sâu sang khoảng 0-255
        depth_norm = cv2.normalize(calibrated_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Thêm text thống kê lên heatmap
        cv2.putText(heatmap, f"Min: {min_val:.2f} m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(heatmap, f"Max: {max_val:.2f} m", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(heatmap, f"Mean: {mean_val:.2f} m", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return heatmap, (min_val, max_val, mean_val)

if __name__ == "__main__":
    # Test module độc lập
    image_path = "src/sample_image.jpg"
    image = Image.open(image_path)

    depth_estimator = DepthEstimator()

    calibrated_depth, output = depth_estimator.infer(image)
    heatmap, stats = depth_estimator.get_heatmap(calibrated_depth)

    print("Output của pipeline:")
    print(output)
    print("Thống kê độ sâu sau hiệu chuẩn (metric):")
    print(f"Min: {stats[0]:.2f} m, Max: {stats[1]:.2f} m, Mean: {stats[2]:.2f} m")

    # Hiển thị ảnh gốc (chuyển từ RGB sang BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("Ảnh gốc", image_cv)
    cv2.imshow("Heatmap (metric depth)", heatmap)
    print("Ấn phím bất kỳ trong cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
