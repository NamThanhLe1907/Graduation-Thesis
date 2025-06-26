"""
Cung cấp công cụ để ước tính độ sâu từ ảnh 2D bằng mô hình Depth Anything.
Hỗ trợ cả Regular Depth và Metric Depth.
Tích hợp camera calibration để cải thiện độ chính xác.
"""
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Union, List, Dict, Any, Optional, Callable

try:
    from .camera_calibration import CameraCalibration
except ImportError:
    # Fallback nếu không import được
    CameraCalibration = None


class DepthEstimator:
    """
    Lớp ước tính độ sâu sử dụng mô hình Depth Anything.
    Hỗ trợ cả Regular Depth và Metric Depth.
    """
    
    # Định nghĩa các model có sẵn
    REGULAR_MODELS = {
        'large': "depth-anything/Depth-Anything-V2-Large-hf",
        'base': "depth-anything/Depth-Anything-V2-Base-hf", 
        'small': "depth-anything/Depth-Anything-V2-Small-hf",
    }
    
    METRIC_MODELS = {
        'indoor_large': "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        'indoor_base': "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
        'indoor_small': "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        'outdoor_large': "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        'outdoor_base': "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
        'outdoor_small': "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    }
    
    def __init__(
        self,
        model_name: str = None,
        model_type: str = "regular",  # "regular" hoặc "metric"
        model_size: str = "small",    # "small", "base", "large"
        scene_type: str = "indoor",   # "indoor" hoặc "outdoor" (chỉ cho metric)
        device: Union[str, torch.device, None] = None,
        fp16: bool = True,
        scale_factor: float = 0.5,
        enable: bool = True,
        input_size: Optional[Tuple[int, int]] = (640, 640),
        skip_frames: int = 20,
        async_mode: bool = True,
        use_camera_calibration: bool = False,
        camera_calibration_file: str = "camera_params.npz",
    ) -> None:
        """
        Khởi tạo ước lượng độ sâu.
        
        Args:
            model_name: Tên mô hình cụ thể từ Hugging Face Hub (nếu None, sẽ tự động chọn)
            model_type: Loại mô hình ("regular" hoặc "metric")
            model_size: Kích thước mô hình ("small", "base", "large")
            scene_type: Loại cảnh ("indoor" hoặc "outdoor") - chỉ áp dụng cho metric
            device: Thiết bị xử lý ('cuda', 'cpu' hoặc None để tự phát hiện)
            fp16: Sử dụng half precision (FP16) nếu có thể
            scale_factor: Hệ số nhân cho độ sâu
            enable: Kích hoạt mô hình (False để tắt hoàn toàn và tiết kiệm tài nguyên)
            input_size: Kích thước resize ảnh đầu vào (W, H), None để giữ nguyên kích thước
            skip_frames: Số frame bỏ qua giữa các lần xử lý (0 = xử lý mọi frame)
            async_mode: Kích hoạt chế độ bất đồng bộ (sử dụng threading)
            use_camera_calibration: Sử dụng camera calibration để undistort ảnh và cải thiện độ chính xác
            camera_calibration_file: Đường dẫn tới file camera calibration (.npz)
        """
        self.model_type = model_type.lower()
        self.model_size = model_size.lower()
        self.scene_type = scene_type.lower()
        
        # Xác định model name nếu không được cung cấp
        if model_name is None:
            model_name = self._get_default_model_name()
        
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
        
        # Khởi tạo camera calibration
        self.camera_calibration = None
        if use_camera_calibration and CameraCalibration is not None:
            try:
                self.camera_calibration = CameraCalibration(camera_calibration_file)
                print(f"[DepthEstimator] Camera calibration đã được tích hợp")
            except Exception as e:
                print(f"[DepthEstimator] Không thể tải camera calibration: {e}")
                self.camera_calibration = None
        elif use_camera_calibration and CameraCalibration is None:
            print(f"[DepthEstimator] Camera calibration không khả dụng (module không được import)")
        
        print(f"[DepthEstimator] Khởi tạo với dtype: {dtype}, device: {self.device}, enable: {enable}")
        print(f"[DepthEstimator] Model type: {self.model_type}")
        print(f"[DepthEstimator] Model: {model_name}")
        if self.model_type == "metric":
            print(f"[DepthEstimator] Scene type: {self.scene_type}")
        if input_size:
            print(f"[DepthEstimator] Resize đầu vào: {input_size}")
        if skip_frames > 0:
            print(f"[DepthEstimator] Bỏ qua {skip_frames} frame giữa các lần xử lý")
        print(f"[DepthEstimator] Chế độ bất đồng bộ: {async_mode}")
        print(f"[DepthEstimator] Camera calibration: {'Bật' if self.camera_calibration else 'Tắt'}")
        
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
    
    def _get_default_model_name(self) -> str:
        """Lấy tên model mặc định dựa trên cấu hình."""
        if self.model_type == "metric":
            # Tạo key cho metric model: scene_size (ví dụ: indoor_small)
            model_key = f"{self.scene_type}_{self.model_size}"
            if model_key in self.METRIC_MODELS:
                return self.METRIC_MODELS[model_key]
            else:
                # Fallback to indoor_small nếu không tìm thấy
                print(f"[DepthEstimator] Không tìm thấy metric model cho {model_key}, sử dụng indoor_small")
                return self.METRIC_MODELS['indoor_small']
        else:
            # Regular model
            if self.model_size in self.REGULAR_MODELS:
                return self.REGULAR_MODELS[self.model_size]
            else:
                # Fallback to small nếu không tìm thấy
                print(f"[DepthEstimator] Không tìm thấy regular model cho {self.model_size}, sử dụng small")
                return self.REGULAR_MODELS['small']
    
    @classmethod
    def create_regular(cls, model_size: str = "small", **kwargs):
        """
        Factory method để tạo regular depth estimator.
        
        Args:
            model_size: Kích thước model ("small", "base", "large")
            **kwargs: Các tham số khác cho __init__
        """
        return cls(model_type="regular", model_size=model_size, **kwargs)
    
    @classmethod 
    def create_metric(cls, scene_type: str = "indoor", model_size: str = "small", **kwargs):
        """
        Factory method để tạo metric depth estimator.
        
        Args:
            scene_type: Loại cảnh ("indoor" hoặc "outdoor")
            model_size: Kích thước model ("small", "base", "large")
            **kwargs: Các tham số khác cho __init__
        """
        return cls(model_type="metric", scene_type=scene_type, model_size=model_size, **kwargs)

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
                - depth_np: Mảng numpy depth map (đơn vị mét cho metric, normalized cho regular)
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
            # Áp dụng camera calibration undistortion nếu có
            processed_image = image
            if self.camera_calibration is not None:
                processed_image = self.camera_calibration.undistort_image(image)
            
            # Giả định ảnh đầu vào là BGR (OpenCV format)
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
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
            # Nội suy lên kích thước gốc - sử dụng phương pháp tương tự code mẫu
            target_size = original_size[::-1]  # Đảo ngược width, height (PIL -> tensor format)
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
                - stats: (min, max, mean) thống kê độ sâu (mét cho metric, normalized cho regular)
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
            unit = "m" if self.model_type == "metric" else "u"  # mét cho metric, unit cho regular
            model_info = f"Model: {self.model_type.upper()}"
            if self.model_type == "metric":
                model_info += f" ({self.scene_type})"
            
            for i, txt in enumerate((
                model_info,
                f"Min: {min_v:.2f} {unit}",
                f"Max: {max_v:.2f} {unit}",
                f"Mean: {mean_v:.2f} {unit}",
            )):
                cv2.putText(
                    heatmap, txt,
                    (10, 30 + i * 30),  # position - giảm khoảng cách
                    font, 0.7, (255, 255, 255), 2, cv2.LINE_AA
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
    
    def estimate_depth_with_3d(self, frame: np.ndarray, bounding_boxes: List[Union[Dict, np.ndarray]]) -> List[Dict]:
        """
        Ước tính độ sâu của các bounding box và chuyển đổi sang tọa độ 3D.
        
        Args:
            frame: Khung hình dạng numpy array (BGR)
            bounding_boxes: Danh sách các bounding box
                
        Returns:
            List[Dict]: Danh sách kết quả với thông tin 3D
                {
                    'bbox': Bounding box gốc,
                    'mean_depth': Độ sâu trung bình,
                    'min_depth': Độ sâu nhỏ nhất,
                    'max_depth': Độ sâu lớn nhất,
                    'center_3d': Tọa độ 3D của center (nếu có camera calibration),
                    'real_size': Kích thước thực tế (nếu có camera calibration)
                }
        """
        # Lấy kết quả depth thông thường
        depth_results = self.estimate_depth(frame, bounding_boxes)
        
        # Nếu không có camera calibration, trả về kết quả thông thường
        if self.camera_calibration is None:
            return depth_results
        
        # Thêm thông tin 3D cho mỗi kết quả
        enhanced_results = []
        for result in depth_results:
            enhanced_result = result.copy()
            bbox = result['bbox']
            mean_depth = result['mean_depth']
            
            if mean_depth > 0:
                # Tính center của bounding box
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Chuyển đổi center sang 3D
                    X, Y, Z = self.camera_calibration.pixel_to_3d(center_x, center_y, mean_depth)
                    enhanced_result['center_3d'] = {
                        'X': X, 'Y': Y, 'Z': Z,
                        'pixel_x': center_x, 'pixel_y': center_y
                    }
                    
                    # Ước tính kích thước thực
                    real_size = self.camera_calibration.estimate_real_size(bbox, mean_depth)
                    enhanced_result['real_size'] = real_size
                else:
                    enhanced_result['center_3d'] = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
                    enhanced_result['real_size'] = {'width_m': 0.0, 'height_m': 0.0, 'area_m2': 0.0}
            else:
                enhanced_result['center_3d'] = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
                enhanced_result['real_size'] = {'width_m': 0.0, 'height_m': 0.0, 'area_m2': 0.0}
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results


if __name__ == "__main__":
    # Thử nghiệm
    import sys
    import os
    
    print("Demo DepthEstimator với Metric Depth")
    print("1. Thử nghiệm Regular Depth")
    print("2. Thử nghiệm Metric Depth (Indoor)")
    print("3. Thử nghiệm Metric Depth (Outdoor)")
    
    choice = input("Chọn loại model (1/2/3): ").strip()
    
    # Tạo đối tượng DepthEstimator dựa trên lựa chọn
    if choice == "1":
        print("Khởi tạo Regular Depth model...")
        depth_estimator = DepthEstimator.create_regular(
            model_size="small", 
            fp16=torch.cuda.is_available()
        )
    elif choice == "2":
        print("Khởi tạo Metric Depth model (Indoor)...")
        depth_estimator = DepthEstimator.create_metric(
            scene_type="indoor", 
            model_size="small", 
            fp16=torch.cuda.is_available()
        )
    elif choice == "3":
        print("Khởi tạo Metric Depth model (Outdoor)...")
        depth_estimator = DepthEstimator.create_metric(
            scene_type="outdoor", 
            model_size="small", 
            fp16=torch.cuda.is_available()
        )
    else:
        print("Lựa chọn không hợp lệ!")
        exit(1)
    
    # Đọc ảnh thử nghiệm
    test_image_path = input("Nhập đường dẫn ảnh thử nghiệm (Enter để dùng sample_image.jpg): ").strip()
    if not test_image_path:
        test_image_path = "sample_image.jpg"
    
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        
        # Giả định có một số bounding box thử nghiệm
        test_boxes = [
            {'bbox': [100, 100, 300, 300]},
            {'bbox': [400, 200, 600, 500]},
        ]
        
        print("Đang xử lý...")
        
        # Ước tính độ sâu
        depth_results = depth_estimator.estimate_depth(frame, test_boxes)
        
        # Tạo và hiển thị depth map dạng heatmap
        depth_map, _ = depth_estimator.infer(frame)
        heatmap, stats = depth_estimator.get_heatmap(depth_map)
        
        # Hiển thị kết quả
        cv2.imshow("Original", frame)
        cv2.imshow("Depth Heatmap", heatmap)
        
        # In thông tin độ sâu
        print(f"\nThống kê depth map:")
        print(f"Min: {stats[0]:.2f}, Max: {stats[1]:.2f}, Mean: {stats[2]:.2f}")
        print(f"Model type: {depth_estimator.model_type}")
        if depth_estimator.model_type == "metric":
            print(f"Scene type: {depth_estimator.scene_type}")
        
        print(f"\nKết quả bounding boxes:")
        for i, result in enumerate(depth_results):
            unit = "m" if depth_estimator.model_type == "metric" else "u"
            print(f"Box {i+1}: Mean depth = {result['mean_depth']:.2f}{unit}")
        
        print("\nNhấn phím bất kỳ để thoát...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Không tìm thấy file ảnh thử nghiệm: {test_image_path}")
        print("Vui lòng đặt file ảnh với tên 'sample_image.jpg' trong thư mục hiện tại hoặc chỉ định đường dẫn khác.") 