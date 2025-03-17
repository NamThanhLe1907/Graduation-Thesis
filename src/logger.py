import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional
import os
import yaml

class AppLogger:
    """Hệ thống logging tập trung với xoay vòng file và giới hạn kích thước"""
    
    _instance: Optional['AppLogger'] = None
    _loggers: Dict[str, logging.Logger] = {}
    
    def __new__(cls, config_path: str = 'configs/camera_settings.yaml'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup(config_path)
        return cls._instance
    
    def _setup(self, config_path: str):
        """Khởi tạo cấu hình logging từ file YAML"""
        with open(config_path) as f:
            config = yaml.safe_load(f)['logging']
            
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._configure_root_logger(config)
        self._configure_file_handler(config)
        
    def _configure_root_logger(self, config: dict):
        logging.basicConfig(
            level=config['level'],
            format=config['format'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _configure_file_handler(self, config: dict):
        file_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, 'app.log'),
            maxBytes=config['file_size_mb'] * 1024 * 1024,
            backupCount=config['max_files'],
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter(config['format']))
        logging.getLogger().addHandler(file_handler)
        
    @classmethod
    def get_logger(cls, name: str = 'root') -> logging.Logger:
        """Lấy logger với tên cụ thể"""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        return cls._loggers[name]

class PerformanceMetrics:
    """Theo dõi hiệu năng hệ thống"""
    
    def __init__(self):
        self.metrics = {
            'frame_processed': 0,
            'avg_processing_time': 0.0,
            'errors': 0
        }
        self.start_time = time.time()
        
    def update(self, processing_time: float, success: bool = True):
        self.metrics['frame_processed'] += 1
        if not success:
            self.metrics['errors'] += 1
            
        # Tính thời gian xử lý trung bình động
        alpha = 0.1  # Hệ số làm mượt
        self.metrics['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.metrics['avg_processing_time']
        )
        
    def report(self):
        """Xuất báo cáo hiệu năng"""
        duration = time.time() - self.start_time
        return {
            'fps': self.metrics['frame_processed'] / duration,
            'avg_processing_ms': self.metrics['avg_processing_time'] * 1000,
            'error_rate': self.metrics['errors'] / self.metrics['frame_processed']
        }