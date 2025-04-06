import cv2
import torch
import numpy as np
from utility.Depth_Anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimatorV2:
    def __init__(self, max_depth=5.75, encoder='vitb', checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vitb.pth', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.max_depth = max_depth
        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth}).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    def infer(self, image):
        """Nhận đầu vào là ảnh OpenCV (BGR numpy array hoặc PIL Image)"""
        # Chuyển đổi sang numpy array và kiểm tra dữ liệu đầu vào
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        if image is None:
            raise ValueError("Ảnh không hợp lệ hoặc không tìm thấy! Vui lòng kiểm tra lại nguồn ảnh.")
            
        # Đảm bảo tính chất của array
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if not image.flags['C_CONTIGUOUS'] or not image.flags['WRITEABLE']:
            image = np.ascontiguousarray(image.copy())
        
        print("Thông số ảnh đầu vào:")
        print("- Type:", type(image))
        print("- Shape:", image.shape)
        print("- Data type:", image.dtype)
        print("- Is contiguous:", image.flags['C_CONTIGUOUS'])
        print("- Is writeable:", image.flags['WRITEABLE'])
        print("- Min/Max values:", np.min(image), np.max(image))
        
        # Nếu ảnh là grayscale, chuyển đổi về BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Kiểm tra lại số kênh
        if image.shape[-1] != 3:
            raise ValueError("Ảnh đầu vào phải có 3 kênh màu (BGR).")
        
        # Kiểm tra tính hợp lệ của ảnh trước khi chuyển đổi
        if image.size == 0:
            raise ValueError("Ảnh đầu vào rỗng")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Ảnh đầu vào phải có 3 kênh màu. Nhận được: {image.shape}")
            
        # Chuyển đổi màu: sử dụng thao tác đảo ngược kênh màu trực tiếp (BGR -> RGB)
        img_rgb = image[..., ::-1].copy()
        
        # Chuyển đổi sang tensor và chuẩn hóa
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            depth = self.model.infer_image(img_tensor)
            if depth.is_cuda:
                depth = depth.detach().cpu()
            else:
                depth = depth.detach()
        return depth.squeeze().numpy().astype(np.float32)

    def get_heatmap(self, depth_map):
        """Tạo heatmap từ depth map"""
        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        mean_val = np.mean(depth_map)
        
        # Chuẩn hóa depth map về khoảng 0-255
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Thêm thông số thống kê lên heatmap
        cv2.putText(heatmap, f"Min: {min_val:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(heatmap, f"Max: {max_val:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(heatmap, f"Mean: {mean_val:.2f}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return heatmap, (min_val, max_val, mean_val)



# if __name__ == "__main__":
    # Test module
    # estimator = DepthEstimatorV2()
    # img = cv2.imread('src/sample_image.jpg')
    # depth = estimator.infer(img)
    # heatmap, _ = estimator.get_heatmap(depth)
    
    # cv2.imshow('Depth Heatmap', heatmap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
