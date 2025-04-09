import cv2
import torch
import numpy as np

# Kiểm tra XFormers
try:
    import xformers
    import xformers.ops
    print("XFormers đã được cài đặt và có thể sử dụng.")

    # Monkey patch scaled_dot_product_attention để dùng xformers
    import torch.nn.functional as F

    def xformers_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        # Bỏ qua mask và dropout để đơn giản
        return xformers.ops.memory_efficient_attention(query, key, value)

    F.scaled_dot_product_attention = xformers_attention
    print("Đã ghi đè scaled_dot_product_attention để dùng xformers.ops.memory_efficient_attention.")
except ImportError:
    print("XFormers chưa được cài đặt hoặc không khả dụng.")

from Depth_Anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitb' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 5.75 # 20 for indoor model, 80 for outdoor model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth}).to(device)
checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vitb.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

raw_img = cv2.imread('utility/sample_image.jpg')

with torch.inference_mode(), torch.amp.autocast('cuda',enabled=(device.type == 'cuda')):
    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy

torch.cuda.empty_cache()

# Tính toán giá trị độ sâu thực tế (cm)
min_depth_cm = float(depth.min() * 100)
max_depth_cm = float(depth.max() * 100)

# Chuẩn hóa depth map để hiển thị và thêm thông số đo lường
depth_normalized = (depth * 255.0 / depth.max()).astype('uint8')
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

# Thêm text hiển thị thông số độ sâu
cv2.putText(depth_colored, f"Min: {min_depth_cm}cm", (10, 30), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(depth_colored, f"Max: {max_depth_cm}cm", (10, 60), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Tạo color bar legend
legend = np.zeros((depth_colored.shape[0], 50, 3), dtype=np.uint8)
cv2.applyColorMap(np.linspace(0, 255, legend.shape[0]).astype(np.uint8), 
                cv2.COLORMAP_INFERNO, legend)
legend = cv2.flip(legend, 0)
cv2.putText(legend, f"{max_depth_cm}cm", (5, 20), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
cv2.putText(legend, f"{min_depth_cm}cm", (5, depth_colored.shape[0]-10), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

# Kết hợp legend và hiển thị
combined = np.hstack([depth_colored, legend])

# Hiển thị ảnh gốc và depth map
cv2.imshow('Original Image', raw_img)
cv2.imshow('Depth Map with Measurements', combined)

# Thêm sự kiện click chuột để xem giá trị depth
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < combined.shape[1] - 50:  # Tránh vùng legend
            depth_cm =float(depth[y, x] * 100 )
            print(f"Depth at ({x}, {y}): {depth_cm}cm")

cv2.setMouseCallback('Depth Map with Measurements', mouse_callback)

cv2.waitKey(0)

cv2.destroyAllWindows()
