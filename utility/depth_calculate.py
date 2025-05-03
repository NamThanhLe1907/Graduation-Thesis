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
        dtype = torch.float16 if (fp16 and self.device.type == "cuda" ) else torch.float32
        
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
        
        inputs = self.processor(images=image, return_tensors= "pt").to(self.device)
        
        #Suy luan
        outputs =  self.model(**inputs)
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