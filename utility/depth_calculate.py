from transformers import AutoImageProcessor , AutoModelForDepthEstimation
import torch 
import torch.nn.functional as F 
import numpy as np 
import cv2 
from PIL import Image 
from typing import Tuple, Union 

class DepthEstimator:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        device: Union[str, torch.device, None] = None,
        fp16: bool = True,
        scale_factor: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"
        dtype = torch.float16 if self.fp16 else torch.float32
        
        print(f"[DepthEstimator] Initializing with dtype: {dtype}, device: {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype = dtype)
            .to(self.device)
            .eval()
        )
        
        self.scale_factor = scale_factor
        
    @torch.inference_mode()
    def infer(
        self,
        image: Image.Image,
        upscale: bool = True,
    )-> Tuple[np.ndarray, torch.Tensor]:
        """
        Parameters
        ----------
        image : PIL.Image
        upscale : bool
            Có nội suy depth map lên kích thước gốc hay không.

        Returns
        -------
        depth_np : np.ndarray  (H, W) — mét
        raw_output : torch.Tensor (predicted_depth) — dạng tensor trước khi nhân scale
        """
        
        #Post_processing
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Chuyển đổi inputs sang đúng kiểu dữ liệu và device như model
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if self.fp16:
                    # Chỉ chuyển sang float16 nếu model dùng fp16
                    inputs[key] = value.to(self.device, dtype=torch.float16)
                else:
                    inputs[key] = value.to(self.device)
        
        # Suy luận với torch.amp.autocast nếu sử dụng fp16
        if self.fp16:
            with torch.amp.autocast('cuda'):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
            
        depth = outputs.predicted_depth # (1,H',W')
        
        if upscale:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size = image.size[::-1],
                mode =  "bicubic",
                align_corners = False,
            ).squeeze(1)
            
            #Chuyen sang CPU, numpy va nhan cho scale_factor
            depth_np = (depth * self.scale_factor).squeeze().cpu().numpy()
            return depth_np, depth # depth (tensor)
        
    def get_heatmap(
        self,
        depth_np:  np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        rgb : bool = False,
        overlay_stats: bool = True,
    )-> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Tạo heat-map màu (BGR) + overlay thống kê.

        Returns
        -------
        heatmap_bgr : np.ndarray
        stats : (min, max, mean) — mét 
        """
        
        min_v, max_v, mean_v = (
            float(depth_np.min()),
            float(depth_np.max()),
            float(depth_np.mean()),
        )
        
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_norm, colormap) #BGR
        
        if overlay_stats:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i , txt in enumerate((
                f"Min: {min_v:.2f} m",
                f"Max: {max_v:.2f} m",
                f"Mean: {mean_v:.2f} m",
            )):
                cv2.putText(
                    heatmap, txt,
                    (10, 30 + i * 40), #position
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
        if rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
          
        return heatmap, (min_v, max_v, mean_v)
    
    def estimate_depth(self, frame, bounding_boxes):
        """
        Calculating the depth of the bounding box in the frame.
        
        Parameters
        ----------
        frame : np.ndarray
        bounding_boxes : list
        
        Returns:
        
        
        
        """
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        depth_map, _ = self.infer(pil_image)
        
        results = []
        
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['bbox']
            
            x1, y1 = max(0,int(x1)), max(0,int(y1))
            x2, y2 = min(int(x2), depth_map.shape[1] -1), min(int(y2), depth_map.shape[0] -1)

            depth_region = depth_map[y1:y2, x1:x2]
            
            if depth_region.size > 0:
                mean_depth = float(np.mean(depth_region))
                min_depth =  float(np.min(depth_region))
                max_depth = float(np.max(depth_region))
            else:
                mean_depth = min_depth = max_depth = 0.0
                
            results.append({
                'bbox': box['bbox'],
                'mean_depth': mean_depth,
                'min_depth': min_depth,
                'max_depth': max_depth,
            })
        return results
    
    
