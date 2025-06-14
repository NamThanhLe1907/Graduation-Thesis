"""
Cung cấp công cụ để ước tính độ sâu từ ảnh 2D bằng mô hình Depth Anything.
"""
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union, List, Dict, Any, Optional, Callable


class DepthEstimator:
    """
    Lớp ước tính độ sâu sử dụng mô hình Depth Anything.
    """
    
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",  # Thay đổi mặc định thành mô hình Small
        device: Union[str, torch.device, None] = None,
        fp16: bool = True,
        scale_factor: float = 1.0,
        enable: bool = True,
        input_size: Optional[Tuple[int, int]] = (640, 640),  # Mặc định kích thước nhỏ
        skip_frames: int = 20,  # Tăng số frame bỏ qua mặc định
        async_mode: bool = True,  # Chế độ bất đồng bộ
    ) -> None:
        """
        Khởi tạo ước lượng độ sâu.
        
        Args:
            model_name: Tên mô hình từ Hugging Face Hub
                - "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" (chất lượng cao, chậm)
                - "depth-anything/Depth-Anything-V2-Base-hf" (nhanh hơn, chất lượng thấp hơn)
                - "depth-anything/Depth-Anything-V2-Small-hf" (nhỏ nhất, nhanh nhất)
            device: Thiết bị xử lý ('cuda', 'cpu' hoặc None để tự phát hiện)
            fp16: Sử dụng half precision (FP16) nếu có thể
            scale_factor: Hệ số nhân cho độ sâu
            enable: Kích hoạt mô hình (False để tắt hoàn toàn và tiết kiệm tài nguyên)
            input_size: Kích thước resize ảnh đầu vào (W, H), None để giữ nguyên kích thước
            skip_frames: Số frame bỏ qua giữa các lần xử lý (0 = xử lý mọi frame)
            async_mode: Kích hoạt chế độ bất đồng bộ (sử dụng threading)
        """
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"
        dtype = torch.float16 if self.fp16 else torch.float32
        self.enable = enable
        self.input_size = input_size
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_depth_map = None
        self.last_depth_tensor = None
        self.async_mode = async_mode
        self.processing = False
        
        print(f"[DepthEstimator] Khởi tạo với dtype: {dtype}, device: {self.device}, enable: {enable}")
        print(f"[DepthEstimator] Model: {model_name}")
        if input_size:
            print(f"[DepthEstimator] Resize đầu vào: {input_size}")
        if skip_frames > 0:
            print(f"[DepthEstimator] Bỏ qua {skip_frames} frame giữa các lần xử lý")
        print(f"[DepthEstimator] Chế độ bất đồng bộ: {async_mode}")
        
        # Tải processor và model nếu được kích hoạt
        if self.enable:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = (
                    AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=dtype)
                    .to(self.device)
                    .eval()
                )
                print(f"[DepthEstimator] Đã tải mô hình thành công: {model_name}")
            except Exception as e:
                raise RuntimeError(f"Lỗi khi tải mô hình Depth Estimation: {str(e)}")
        else:
            print(f"[DepthEstimator] Mô hình bị tắt, sẽ không sử dụng tài nguyên")
            self.processor = None
            self.model = None
        
        self.scale_factor = scale_factor
    
    @torch.inference_mode()
    def infer(
        self,
        image: Union[Image.Image, np.ndarray],
        upscale: bool = True,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Thực hiện suy luận độ sâu trên một ảnh.
        
        Args:
            image: Ảnh PIL hoặc mảng numpy (BGR hoặc RGB)
            upscale: Nội suy depth map lên kích thước gốc hay không
            
        Returns:
            Tuple: (depth_np, raw_output)
                - depth_np: Mảng numpy depth map đơn vị mét
                - raw_output: Tensor depth map gốc
        """
        # Nếu model bị tắt, trả về kết quả trống
        if not self.enable:
            if isinstance(image, np.ndarray):
                # Trả về depth map toàn 0 với cùng kích thước ảnh đầu vào
                h, w = image.shape[:2]
                depth_np = np.zeros((h, w), dtype=np.float32)
                depth_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                return depth_np, depth_tensor
            elif isinstance(image, Image.Image):
                w, h = image.size
                depth_np = np.zeros((h, w), dtype=np.float32)
                depth_tensor = torch.zeros((1, h, w), dtype=torch.float32)
                return depth_np, depth_tensor
        
        # Xử lý định kỳ - bỏ qua một số frame
        self.frame_count += 1
        if self.skip_frames > 0 and (self.frame_count - 1) % (self.skip_frames + 1) != 0:
            # Trả về kết quả của lần xử lý trước đó nếu có
            if self.last_depth_map is not None and self.last_depth_tensor is not None:
                return self.last_depth_map, self.last_depth_tensor
        
        # Chuyển đổi từ numpy sang PIL Image nếu cần
        if isinstance(image, np.ndarray):
            # Giả định ảnh đầu vào là BGR (OpenCV format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image
        
        # Resize ảnh đầu vào nếu có chỉ định kích thước
        original_size = pil_image.size
        if self.input_size:
            pil_image = pil_image.resize(self.input_size, Image.BILINEAR)
        
        # Xử lý đầu vào
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Chuyển dữ liệu lên thiết bị đúng
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if self.fp16:
                    # Chỉ chuyển sang float16 nếu model dùng fp16
                    inputs[key] = value.to(self.device, dtype=torch.float16)
                else:
                    inputs[key] = value.to(self.device)
        
        # Thực hiện suy luận với cấu hình phù hợp
        if self.fp16:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
            
        depth = outputs.predicted_depth  # (1, H', W')
        
        if upscale:
            # Nội suy lên kích thước gốc
            target_size = original_size[::-1]  # Đảo ngược width, height
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=target_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            
            # Chuyển sang CPU, numpy và nhân với scale_factor
            depth_np = (depth * self.scale_factor).squeeze().cpu().numpy()
            
            # Lưu kết quả cho lần sau
            self.last_depth_map = depth_np
            self.last_depth_tensor = depth
            
            return depth_np, depth  # depth_np (numpy), depth (tensor)
        else:
            # Không nội suy
            depth_np = (depth * self.scale_factor).squeeze().cpu().numpy()
            
            # Lưu kết quả cho lần sau
            self.last_depth_map = depth_np
            self.last_depth_tensor = depth
            
            return depth_np, depth
    
    @torch.inference_mode()
    def infer_async(self, 
               image: Union[Image.Image, np.ndarray], 
               callback: Optional[Callable[[np.ndarray, torch.Tensor], None]] = None,
               upscale: bool = True) -> bool:
        """
        Thực hiện suy luận độ sâu bất đồng bộ trên một ảnh.
        
        Args:
            image: Ảnh PIL hoặc mảng numpy (BGR hoặc RGB)
            callback: Hàm callback được gọi khi hoàn thành với (depth_np, depth_tensor)
            upscale: Nội suy depth map lên kích thước gốc hay không
            
        Returns:
            bool: True nếu xử lý được bắt đầu, False nếu không thể bắt đầu xử lý
        """
        if not self.enable:
            # Không xử lý nếu model bị tắt
            return False
            
        if self.processing:
            # Đang xử lý, không thể bắt đầu một xử lý mới
            return False
            
        # Xử lý định kỳ - bỏ qua một số frame
        self.frame_count += 1
        if self.skip_frames > 0 and (self.frame_count - 1) % (self.skip_frames + 1) != 0:
            # Trả về kết quả của lần xử lý trước đó nếu có
            if self.last_depth_map is not None and self.last_depth_tensor is not None and callback:
                callback(self.last_depth_map, self.last_depth_tensor)
            return False
        
        if not self.async_mode:
            # Nếu không ở chế độ bất đồng bộ, xử lý đồng bộ và gọi callback
            depth_np, depth_tensor = self.infer(image, upscale)
            if callback:
                callback(depth_np, depth_tensor)
            return True
            
        # Xử lý bất đồng bộ bằng thread
        import threading
        
        def process_thread():
            try:
                self.processing = True
                depth_np, depth_tensor = self.infer(image, upscale)
                if callback:
                    callback(depth_np, depth_tensor)
            finally:
                self.processing = False
                
        # Khởi động thread xử lý
        threading.Thread(target=process_thread, daemon=True).start()
        return True
    
    def get_heatmap(
        self,
        depth_np: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        rgb: bool = False,
        overlay_stats: bool = True,
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tạo heatmap màu từ depth map.
        
        Args:
            depth_np: Depth map dạng numpy array
            colormap: Loại colormap của OpenCV
            rgb: Trả về ảnh RGB thay vì BGR
            overlay_stats: Hiển thị thông tin thống kê trên heatmap
            
        Returns:
            Tuple: (heatmap, stats)
                - heatmap: Ảnh heatmap (BGR hoặc RGB)
                - stats: (min, max, mean) thống kê độ sâu (mét)
        """
        # Tính toán thống kê
        min_v, max_v, mean_v = (
            float(depth_np.min()),
            float(depth_np.max()),
            float(depth_np.mean()),
        )
        
        # Chuẩn hóa depth map về khoảng [0, 255]
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Áp dụng colormap
        heatmap = cv2.applyColorMap(depth_norm, colormap)  # BGR
        
        # Thêm thông tin thống kê
        if overlay_stats:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, txt in enumerate((
                f"Min: {min_v:.2f} m",
                f"Max: {max_v:.2f} m",
                f"Mean: {mean_v:.2f} m",
            )):
                cv2.putText(
                    heatmap, txt,
                    (10, 30 + i * 40),  # position
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
                
        # Chuyển đổi sang RGB nếu cần
        if rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap, (min_v, max_v, mean_v)
    
    def estimate_depth(self, frame: np.ndarray, bounding_boxes: List[Union[Dict, np.ndarray]]) -> List[Dict]:
        """
        Ước tính độ sâu của các bounding box trong khung hình.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            bounding_boxes: Danh sách các bounding box
                Mỗi phần tử có thể là:
                - Dict với khóa 'bbox' chứa [x1, y1, x2, y2]
                - Hoặc mảng [x1, y1, x2, y2]
                
        Returns:
            List[Dict]: Danh sách kết quả cho từng bounding box
                {
                    'bbox': Bounding box gốc,
                    'mean_depth': Độ sâu trung bình,
                    'min_depth': Độ sâu nhỏ nhất,
                    'max_depth': Độ sâu lớn nhất
                }
        """
        # Nếu model bị tắt, trả về kết quả trống
        if not self.enable:
            results = []
            for box in bounding_boxes:
                # Xác định định dạng bounding box
                if isinstance(box, dict) and 'bbox' in box:
                    original_box = box['bbox']
                elif isinstance(box, (list, np.ndarray)) and len(box) >= 4:
                    original_box = box
                else:
                    continue
                
                # Trả về kết quả giả với độ sâu bằng 0
                results.append({
                    'bbox': original_box,
                    'mean_depth': 0.0,
                    'min_depth': 0.0,
                    'max_depth': 0.0,
                })
            return results
        
        # Chuyển sang RGB và tạo PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Thực hiện suy luận để có depth map
        depth_map, _ = self.infer(pil_image)
        
        results = []
        
        for box in bounding_boxes:
            # Xác định định dạng bounding box
            if isinstance(box, dict) and 'bbox' in box:
                x1, y1, x2, y2 = box['bbox']
                original_box = box['bbox']
            elif isinstance(box, (list, np.ndarray)) and len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                original_box = box
            else:
                # Bỏ qua bounding box không hợp lệ
                continue
            
            # Giới hạn các tọa độ trong phạm vi của depth map
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(int(x2), depth_map.shape[1] - 1), min(int(y2), depth_map.shape[0] - 1)
            
            # Trích xuất vùng độ sâu cho bounding box
            depth_region = depth_map[y1:y2, x1:x2]
            
            # Tính toán thống kê độ sâu
            if depth_region.size > 0:
                mean_depth = float(np.mean(depth_region))
                min_depth = float(np.min(depth_region))
                max_depth = float(np.max(depth_region))
            else:
                mean_depth = min_depth = max_depth = 0.0
            
            # Đóng gói kết quả
            results.append({
                'bbox': original_box,
                'mean_depth': mean_depth,
                'min_depth': min_depth,
                'max_depth': max_depth,
            })
            
        return results


if __name__ == "__main__":
    # Thử nghiệm
    import sys
    import os
    
    # Tạo đối tượng DepthEstimator
    depth_estimator = DepthEstimator(fp16=torch.cuda.is_available())
    
    # Đọc ảnh thử nghiệm
    test_image_path = "sample_image.jpg"
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        
        # Giả định có một số bounding box thử nghiệm
        test_boxes = [
            {'bbox': [100, 100, 300, 300]},
            {'bbox': [400, 200, 600, 500]},
        ]
        
        # Ước tính độ sâu
        depth_results = depth_estimator.estimate_depth(frame, test_boxes)
        
        # Tạo và hiển thị depth map dạng heatmap
        depth_map, _ = depth_estimator.infer(frame)
        heatmap, stats = depth_estimator.get_heatmap(depth_map)
        
        # Hiển thị kết quả
        cv2.imshow("Original", frame)
        cv2.imshow("Depth Heatmap", heatmap)
        
        # In thông tin độ sâu
        for i, result in enumerate(depth_results):
            print(f"Box {i+1}: Mean depth = {result['mean_depth']:.2f}m")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Không tìm thấy file ảnh thử nghiệm: {test_image_path}") 