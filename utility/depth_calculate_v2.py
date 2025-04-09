import cv2
import torch
import numpy as np

# Kiểm tra và patch XFormers
try:
    import xformers
    import xformers.ops
    print("XFormers đã được cài đặt và có thể sử dụng.")

    import torch.nn.functional as F

    def xformers_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        # Bỏ qua mask và dropout để đơn giản
        return xformers.ops.memory_efficient_attention(query, key, value)

    F.scaled_dot_product_attention = xformers_attention
    print("Đã ghi đè scaled_dot_product_attention để dùng xformers.ops.memory_efficient_attention.")
except ImportError:
    print("XFormers chưa được cài đặt hoặc không khả dụng.")

from utility.Depth_Anything.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimatorV2:
    def __init__(self, max_depth=0.6, encoder='vitb', checkpoint_path='utility/checkpoint/depth_anything_v2_metric_hypersim_vitb.pth', device='cuda'):
        """
        max_depth: giá trị tối đa (tính bằng mét) mà mô hình dự đoán, ví dụ 1.2 cho 120cm
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Kiểm tra XFormers
        try:
            import xformers
            self.use_xformers = True
        except ImportError:
            self.use_xformers = False

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        self.max_depth = max_depth  # đơn vị mét
        
        try:
            self.model = DepthAnythingV2(
                **{**model_configs[encoder], 'max_depth': max_depth}
            ).to(self.device)
            
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def infer(self, batch_tensor):
        """Nhận batch tensor (B, 3, H, W) đã chuẩn hóa"""
        with torch.no_grad():
            try:
                depths = self.model.infer_image(batch_tensor)
                return depths.detach()
            except Exception as e:
                torch.cuda.empty_cache()
                raise RuntimeError(f"Inference failed: {str(e)}")

    def get_heatmap(self, depth_map, unit='cm'):
        """
        Tạo heatmap từ depth map với đơn vị mong muốn.
        unit: 'm', 'cm', hoặc 'mm'
        """
        # Nếu depth_map là Tensor, convert sang numpy
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.detach().cpu().numpy()
        elif not isinstance(depth_map, np.ndarray):
            try:
                depth_map = np.array(depth_map)
            except:
                raise TypeError(f"depth_map must be a numpy array or tensor, but got {type(depth_map)}")

        # Chuyển đổi đơn vị
        if unit == 'cm':
            depth_map = depth_map * 100
        elif unit == 'mm':
            depth_map = depth_map * 1000
        # nếu 'm' thì giữ nguyên

        min_val = np.min(depth_map)
        max_val = np.max(depth_map)
        mean_val = np.mean(depth_map)
        
        # Chuẩn hóa depth map về khoảng 0-255 để hiển thị màu
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        
        # Thêm thông số thống kê lên heatmap, kèm đơn vị
        cv2.putText(heatmap, f"Min: {min_val:.1f} {unit}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(heatmap, f"Max: {max_val:.1f} {unit}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(heatmap, f"Mean: {mean_val:.1f} {unit}", (10, 110), 
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
