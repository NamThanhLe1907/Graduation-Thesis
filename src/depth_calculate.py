from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

# Khởi tạo pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", use_fast=True)

# Đọc ảnh cục bộ
image_path = "src/sample_image.jpg"
image = Image.open(image_path)

# Suy luận độ sâu
output = pipe(image)
pred_depth_np = np.array(output["predicted_depth"])

# Giả sử bạn đã tính scale_factor từ quá trình hiệu chuẩn (ví dụ: 1.1)
scale_factor = 0.325 # Thay bằng giá trị thực tế từ hiệu chuẩn
calibrated_depth = pred_depth_np * scale_factor

# Tính thống kê độ sâu sau hiệu chuẩn
min_val = calibrated_depth.min()
max_val = calibrated_depth.max()
mean_val = calibrated_depth.mean()
print("Thống kê độ sâu sau hiệu chuẩn (metric):")
print(f"Min: {min_val:.2f} m, Max: {max_val:.2f} m, Mean: {mean_val:.2f} m")

# Chuẩn hóa để hiển thị
depth_norm = cv2.normalize(calibrated_depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)
heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# Thêm text thống kê lên heatmap
cv2.putText(heatmap, f"Min: {min_val:.2f} m", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(heatmap, f"Max: {max_val:.2f} m", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(heatmap, f"Mean: {mean_val:.2f} m", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Hiển thị
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
cv2.imshow("Ảnh gốc", image_cv)
cv2.imshow("Heatmap (metric depth)", heatmap)
print("Ấn phím bất kỳ trong cửa sổ ảnh để thoát...")
cv2.waitKey(0)
cv2.destroyAllWindows()