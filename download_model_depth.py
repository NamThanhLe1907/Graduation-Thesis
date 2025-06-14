#!/usr/bin/env python3
"""
Script để tải trước các model Depth Anything V2 từ Hugging Face Hub
"""

import os
import sys
import time
import shutil
import traceback
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Định nghĩa các model cần tải
MODELS_TO_DOWNLOAD = {
    # Metric Models cho Indoor (theo cấu hình hiện tại của bạn)
    "indoor_large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "indoor_base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", 
    "indoor_small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    
    # Nếu bạn muốn thêm Outdoor models
    # "outdoor_large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    # "outdoor_base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    # "outdoor_small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    
    # Nếu bạn muốn thêm Regular models
    # "regular_large": "depth-anything/Depth-Anything-V2-Large-hf",
    # "regular_base": "depth-anything/Depth-Anything-V2-Base-hf",
    # "regular_small": "depth-anything/Depth-Anything-V2-Small-hf",
}

def download_model(model_name: str, model_id: str, force_download: bool = False):
    """
    Tải một model từ Hugging Face Hub
    
    Args:
        model_name: Tên model để hiển thị
        model_id: ID model trên Hugging Face Hub
        force_download: Ép buộc tải lại nếu đã có
    """
    print(f"\n{'='*60}")
    print(f"Đang tải model: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}")
    
    try:
        # Tải processor trước
        print("🔄 Đang tải Image Processor...")
        processor = AutoImageProcessor.from_pretrained(
            model_id, 
            force_download=force_download
        )
        print("✅ Image Processor đã được tải thành công!")
        
        # Tải model
        print("🔄 Đang tải Model...")
        
        # Sử dụng torch.float32 để tránh lỗi với một số GPU
        model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Sử dụng float32 để tương thích tốt hơn
            force_download=force_download
        )
        print("✅ Model đã được tải thành công!")
        
        # Kiểm tra kích thước model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Số lượng tham số: {total_params:,}")
        
        # Kiểm tra memory footprint
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
        print(f"💾 Kích thước model: {model_size_mb:.1f} MB")
        
        # Test nhanh để đảm bảo model hoạt động
        print("🧪 Đang kiểm tra model...")
        
        # Tạo ảnh test giả từ numpy
        import numpy as np
        from PIL import Image
        
        # Tạo ảnh RGB test 224x224
        test_image_np = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image_np)
        
        # Xử lý với processor
        test_processed = processor(
            images=test_image_pil, 
            return_tensors="pt"
        )
        
        # Thực hiện inference test
        with torch.no_grad():
            test_output = model(**test_processed)
            
        print("✅ Model đã được kiểm tra và hoạt động tốt!")
        print(f"📏 Kích thước output: {test_output.predicted_depth.shape}")
        
        # Dọn dẹp memory
        del model, processor, test_image_np, test_image_pil, test_processed, test_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tải model {model_name}:")
        print(f"   {str(e)}")
        traceback.print_exc()
        return False

def check_system_requirements():
    """Kiểm tra yêu cầu hệ thống"""
    print("🔍 Kiểm tra yêu cầu hệ thống...")
    
    # Kiểm tra Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Kiểm tra torch
    try:
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"❌ PyTorch lỗi: {e}")
        return False
    
    # Kiểm tra transformers
    try:
        import transformers
        print(f"🤗 Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers chưa được cài đặt!")
        return False
    
    # Kiểm tra disk space
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        total, used, free = shutil.disk_usage(cache_dir)
        print(f"💽 Disk space available: {free / 1024**3:.1f} GB")
        if free < 10 * 1024**3:  # 10GB
            print("⚠️  Cảnh báo: Ít hơn 10GB dung lượng trống!")
    
    return True

def main():
    print("🚀 Bắt đầu tải các model Depth Anything V2")
    print("="*60)
    
    # Kiểm tra yêu cầu hệ thống
    if not check_system_requirements():
        print("❌ Yêu cầu hệ thống không đáp ứng!")
        return
    
    # Hỏi người dùng có muốn force download không
    force_download = False
    response = input("\n🤔 Bạn có muốn tải lại các model đã có sẵn không? (y/N): ").strip().lower()
    if response in ['y', 'yes', 'có']:
        force_download = True
    
    # Bắt đầu tải models
    success_count = 0
    total_count = len(MODELS_TO_DOWNLOAD)
    
    for model_name, model_id in MODELS_TO_DOWNLOAD.items():
        success = download_model(model_name, model_id, force_download)
        if success:
            success_count += 1
        
        # Pause giữa các lần tải để giảm tải server
        if model_name != list(MODELS_TO_DOWNLOAD.keys())[-1]:  # Không pause ở lần cuối
            print("\n⏸️  Tạm dừng 5 giây trước khi tải model tiếp...")
            time.sleep(5)
    
    # Tóm tắt kết quả
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ TỔNG QUAN")
    print(f"{'='*60}")
    print(f"✅ Thành công: {success_count}/{total_count} models")
    print(f"❌ Thất bại: {total_count - success_count}/{total_count} models")
    
    if success_count == total_count:
        print("\n🎉 Tất cả models đã được tải thành công!")
        print("💡 Bây giờ bạn có thể chạy use_tensorrt_example.py mà không cần tải model online.")
        
        print("\n📋 Các model đã tải:")
        for model_name, model_id in MODELS_TO_DOWNLOAD.items():
            print(f"   • {model_name}: {model_id}")
            
        print("\n🛠️  Cách sử dụng:")
        print("   # Sử dụng Large model (mặc định)")
        print("   python use_tensorrt_example.py")
        print("   ")
        print("   # Sử dụng Base model")
        print("   set DEPTH_MODEL=base && python use_tensorrt_example.py")
        print("   ")
        print("   # Sử dụng Small model (nhanh nhất)")
        print("   set DEPTH_MODEL=small && python use_tensorrt_example.py")
        
    else:
        print(f"\n⚠️  Một số models không tải được. Vui lòng kiểm tra lại kết nối mạng và thử lại.")
    
    print(f"\n📁 Models được lưu tại: {os.path.expanduser('~/.cache/huggingface/transformers')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Đã hủy bỏ quá trình tải!")
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {str(e)}")
        traceback.print_exc()