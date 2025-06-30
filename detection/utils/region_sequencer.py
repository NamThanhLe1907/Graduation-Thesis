"""
Region Sequencer cho sequential region extraction & placement.
Quản lý thứ tự gửi regions theo sequence P1R1 → P1R3 → P1R2 (bao giữa luôn cuối cùng).
"""
import time
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RegionSequencer:
    """
    Class quản lý sequence placement cho regions.
    Default sequence: [1, 3, 2] - bao giữa (region 2) luôn cuối cùng.
    """
    
    def __init__(self, sequence: List[int] = [1, 3, 2]):
        """
        Khởi tạo RegionSequencer.
        
        Args:
            sequence: Thứ tự regions cần gửi (default: [1, 3, 2])
        """
        self.sequence = sequence
        self.placement_queue = []
        self.current_index = 0
        self.current_pallet_id = None
        self.status = "IDLE"  # IDLE, SENDING, WAITING, COMPLETED
        
    def add_pallet_to_queue(self, pallet_regions: List[Dict], pallet_id: int = 1):
        """
        Thêm pallet regions vào queue theo sequence [1, 3, 2].
        
        Args:
            pallet_regions: List regions của pallet
            pallet_id: ID của pallet
        """
        logger.info(f"[SEQUENCER] Adding pallet {pallet_id} with {len(pallet_regions)} regions to queue")
        
        # Lọc và sắp xếp regions theo sequence
        ordered_regions = []
        for seq_id in self.sequence:
            # Tìm region với region_id = seq_id
            matching_region = None
            for region in pallet_regions:
                region_info = region.get('region_info', {})
                if region_info.get('region_id') == seq_id:
                    matching_region = region.copy()
                    break
            
            if matching_region:
                # Thêm metadata cho placement
                matching_region['sequence_order'] = len(ordered_regions) + 1
                matching_region['status'] = 'pending'
                matching_region['pallet_id'] = pallet_id
                matching_region['added_time'] = time.time()
                ordered_regions.append(matching_region)
                
                logger.debug(f"[SEQUENCER] Added Region {seq_id} as sequence position {len(ordered_regions)}")
            else:
                logger.warning(f"[SEQUENCER] Region {seq_id} not found in pallet {pallet_id}")
        
        # Thêm vào queue
        self.placement_queue.extend(ordered_regions)
        self.current_pallet_id = pallet_id
        
        logger.info(f"[SEQUENCER] Queue updated: {len(self.placement_queue)} regions total")
        
    def get_next_region(self) -> Optional[Dict]:
        """
        Lấy region tiếp theo từ queue.
        
        Returns:
            Dict region data hoặc None nếu queue empty
        """
        if self.current_index >= len(self.placement_queue):
            return None
        
        region = self.placement_queue[self.current_index]
        region['status'] = 'sending'
        self.status = "SENDING"
        
        region_info = region.get('region_info', {})
        region_id = region_info.get('region_id', 'Unknown')
        sequence_order = region.get('sequence_order', 'Unknown')
        
        logger.info(f"[SEQUENCER] Next region: P{self.current_pallet_id}R{region_id} (sequence {sequence_order}/{len(self.sequence)})")
        
        return region
    
    def mark_region_completed(self):
        """
        Đánh dấu region hiện tại đã hoàn thành và chuyển sang region tiếp theo.
        """
        if self.current_index < len(self.placement_queue):
            region = self.placement_queue[self.current_index]
            region['status'] = 'completed'
            region['completed_time'] = time.time()
            
            region_info = region.get('region_info', {})
            region_id = region_info.get('region_id', 'Unknown')
            
            logger.info(f"[SEQUENCER] ✅ Completed P{self.current_pallet_id}R{region_id}")
            
            self.current_index += 1
            
            if self.current_index >= len(self.placement_queue):
                self.status = "COMPLETED"
                logger.info(f"[SEQUENCER] ✅ All regions completed for pallet {self.current_pallet_id}")
            else:
                self.status = "WAITING"
                logger.info(f"[SEQUENCER] ⏳ Waiting for next region...")
    
    def get_queue_status(self) -> Dict:
        """
        Lấy status của queue.
        
        Returns:
            Dict chứa thông tin status
        """
        remaining_regions = []
        completed_regions = []
        
        for i, region in enumerate(self.placement_queue):
            region_info = region.get('region_info', {})
            region_data = {
                'region_id': region_info.get('region_id'),
                'sequence_order': region.get('sequence_order'),
                'status': region.get('status'),
                'is_current': i == self.current_index
            }
            
            if region.get('status') == 'completed':
                completed_regions.append(region_data)
            else:
                remaining_regions.append(region_data)
        
        return {
            'status': self.status,
            'current_pallet': self.current_pallet_id,
            'current_index': self.current_index,
            'total_regions': len(self.placement_queue),
            'completed_count': len(completed_regions),
            'remaining_count': len(remaining_regions),
            'progress': f"{len(completed_regions)}/{len(self.placement_queue)}",
            'completed_regions': completed_regions,
            'remaining_regions': remaining_regions,
            'sequence': self.sequence
        }
    
    def reset_sequence(self):
        """
        Reset sequence về trạng thái ban đầu.
        """
        logger.info("[SEQUENCER] Resetting sequence")
        
        self.placement_queue.clear()
        self.current_index = 0
        self.current_pallet_id = None
        self.status = "IDLE"
    
    def skip_current_region(self):
        """
        Bỏ qua region hiện tại và chuyển sang region tiếp theo.
        """
        if self.current_index < len(self.placement_queue):
            region = self.placement_queue[self.current_index]
            region['status'] = 'skipped'
            
            region_info = region.get('region_info', {})
            region_id = region_info.get('region_id', 'Unknown')
            
            logger.info(f"[SEQUENCER] ⏭️ Skipped P{self.current_pallet_id}R{region_id}")
            
            self.current_index += 1
            
            if self.current_index >= len(self.placement_queue):
                self.status = "COMPLETED"
            else:
                self.status = "WAITING"
    
    def get_current_region(self) -> Optional[Dict]:
        """
        Lấy region hiện tại đang được xử lý.
        
        Returns:
            Dict region data hoặc None
        """
        if self.current_index < len(self.placement_queue):
            return self.placement_queue[self.current_index]
        return None
    
    def is_queue_empty(self) -> bool:
        """
        Kiểm tra queue có empty không.
        
        Returns:
            True nếu queue empty
        """
        return len(self.placement_queue) == 0
    
    def is_sequence_completed(self) -> bool:
        """
        Kiểm tra sequence đã hoàn thành chưa.
        
        Returns:
            True nếu sequence đã hoàn thành
        """
        return self.status == "COMPLETED" 