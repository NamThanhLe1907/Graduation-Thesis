"""
🚀 REGION SEQUENCER STATUS TRIGGERING EXAMPLE
Giải thích cách trigger và handle status changes trong RegionSequencer
"""
import time
from typing import Callable, Optional, Dict, Any

# ===========================================================================
# 📊 STATUS LIFECYCLE EXPLANATION
# ===========================================================================

"""
CURRENT STATUS FLOW:
IDLE → SENDING → WAITING → COMPLETED
  ↓       ↓        ↓         ↓
 Reset   Send    Robot    All Done
        Next   Complete

Hiện tại RegionSequencer CHỈ CÓ internal status tracking.
Để trigger actions, cần ADD CALLBACK SYSTEM.
"""

# ===========================================================================
# ⭐ ENHANCED REGION SEQUENCER WITH CALLBACKS
# ===========================================================================

class EnhancedRegionSequencer:
    """
    Enhanced RegionSequencer với callback system để trigger actions.
    """
    
    def __init__(self, 
                 sequence=[1, 3, 2],
                 on_status_change: Optional[Callable[[str, str], None]] = None,
                 on_region_ready: Optional[Callable[[Dict], None]] = None,
                 on_region_sent: Optional[Callable[[Dict], None]] = None,
                 on_sequence_completed: Optional[Callable[[], None]] = None):
        """
        Args:
            on_status_change: Callback(old_status, new_status) - Khi status thay đổi
            on_region_ready: Callback(region_data) - Khi có region ready để gửi  
            on_region_sent: Callback(region_data) - Khi region đã được gửi
            on_sequence_completed: Callback() - Khi sequence hoàn thành
        """
        self.sequence = sequence
        self.placement_queue = []
        self.current_index = 0
        self.current_pallet_id = None
        self._status = "IDLE"
        
        # ⭐ CALLBACK FUNCTIONS ⭐
        self.on_status_change = on_status_change
        self.on_region_ready = on_region_ready
        self.on_region_sent = on_region_sent
        self.on_sequence_completed = on_sequence_completed
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, new_status):
        """Trigger callback khi status thay đổi"""
        old_status = self._status
        self._status = new_status
        
        # ⭐ TRIGGER STATUS CHANGE CALLBACK ⭐
        if self.on_status_change and old_status != new_status:
            self.on_status_change(old_status, new_status)
    
    def add_pallet_to_queue(self, pallet_regions, pallet_id=1):
        """Add pallet với callback triggers"""
        print(f"[SEQUENCER] Adding pallet {pallet_id} to queue...")
        
        # Logic như cũ...
        ordered_regions = []
        for seq_id in self.sequence:
            matching_region = None
            for region in pallet_regions:
                region_info = region.get('region_info', {})
                if region_info.get('region_id') == seq_id:
                    matching_region = region.copy()
                    break
            
            if matching_region:
                matching_region['sequence_order'] = len(ordered_regions) + 1
                matching_region['status'] = 'pending'
                matching_region['pallet_id'] = pallet_id
                matching_region['added_time'] = time.time()
                ordered_regions.append(matching_region)
        
        self.placement_queue.extend(ordered_regions)
        self.current_pallet_id = pallet_id
        
        # ⭐ TRIGGER: First region ready ⭐
        if len(ordered_regions) > 0 and self.on_region_ready:
            self.on_region_ready(ordered_regions[0])
        
        # Status change từ IDLE → WAITING (có regions trong queue)
        if self.status == "IDLE" and len(self.placement_queue) > 0:
            self.status = "WAITING"
    
    def get_next_region(self):
        """Get next region với callbacks"""
        if self.current_index >= len(self.placement_queue):
            return None
        
        region = self.placement_queue[self.current_index]
        region['status'] = 'sending'
        
        # ⭐ TRIGGER STATUS CHANGE: WAITING/IDLE → SENDING ⭐
        self.status = "SENDING"
        
        # ⭐ TRIGGER: Region được gửi ⭐
        if self.on_region_sent:
            self.on_region_sent(region)
        
        return region
    
    def mark_region_completed(self):
        """Mark completed với callbacks"""
        if self.current_index < len(self.placement_queue):
            region = self.placement_queue[self.current_index]
            region['status'] = 'completed'
            region['completed_time'] = time.time()
            
            self.current_index += 1
            
            if self.current_index >= len(self.placement_queue):
                # ⭐ TRIGGER STATUS CHANGE: SENDING → COMPLETED ⭐
                self.status = "COMPLETED"
                
                # ⭐ TRIGGER: Sequence hoàn thành ⭐
                if self.on_sequence_completed:
                    self.on_sequence_completed()
            else:
                # ⭐ TRIGGER STATUS CHANGE: SENDING → WAITING ⭐
                self.status = "WAITING"
                
                # ⭐ TRIGGER: Next region ready ⭐
                if self.on_region_ready:
                    next_region = self.placement_queue[self.current_index]
                    self.on_region_ready(next_region)

# ===========================================================================
# 🎯 CALLBACK FUNCTIONS - DEFINE YOUR ACTIONS HERE
# ===========================================================================

def on_status_changed(old_status: str, new_status: str):
    """Called when status changes"""
    print(f"📊 [STATUS] {old_status} → {new_status}")
    
    # ⭐ TRIGGER ACTIONS BASED ON STATUS ⭐
    if new_status == "SENDING":
        print("🚀 [ACTION] Starting to send region to PLC...")
        # Trigger PLC sending logic here
        
    elif new_status == "WAITING":
        print("⏳ [ACTION] Waiting for robot completion signal...")
        # Setup robot monitoring here
        
    elif new_status == "COMPLETED":
        print("🎉 [ACTION] All regions completed! Cleanup resources...")
        # Cleanup, reset, or start new cycle
        
    elif new_status == "IDLE":
        print("😴 [ACTION] System idle, ready for new pallets...")

def on_region_ready(region_data: Dict):
    """Called when region is ready to be sent"""
    region_info = region_data.get('region_info', {})
    region_id = region_info.get('region_id', 'Unknown')
    pallet_id = region_data.get('pallet_id', 'Unknown')
    
    print(f"📦 [READY] P{pallet_id}R{region_id} is ready for sending")
    
    # ⭐ TRIGGER: Prepare PLC data ⭐
    # prepare_plc_data(region_data)

def on_region_sent(region_data: Dict):
    """Called when region is being sent"""
    region_info = region_data.get('region_info', {})
    region_id = region_info.get('region_id', 'Unknown') 
    pallet_id = region_data.get('pallet_id', 'Unknown')
    
    print(f"📡 [SENT] P{pallet_id}R{region_id} sending to PLC...")
    
    # ⭐ TRIGGER: Send to PLC ⭐
    # send_to_plc(region_data)
    # start_robot_monitoring()

def on_sequence_completed():
    """Called when entire sequence is completed"""
    print("✅ [COMPLETED] Entire sequence finished!")
    
    # ⭐ TRIGGER: Cleanup actions ⭐
    # cleanup_resources()
    # send_completion_signal()
    # reset_for_next_pallet()

# ===========================================================================
# 🧪 DEMO USAGE - HIỂN THỊ CÁCH SỬ DỤNG CALLBACK SYSTEM  
# ===========================================================================

def demo_callback_system():
    """Demo enhanced sequencer với callback system"""
    print("🚀 DEMO: Enhanced RegionSequencer with Callback System")
    print("=" * 60)
    
    # ⭐ TẠO SEQUENCER VỚI CALLBACKS ⭐
    sequencer = EnhancedRegionSequencer(
        sequence=[1, 3, 2],
        on_status_change=on_status_changed,
        on_region_ready=on_region_ready,
        on_region_sent=on_region_sent,
        on_sequence_completed=on_sequence_completed
    )
    
    # Mock pallet regions
    mock_regions = [
        {
            'region_info': {'region_id': 1, 'pallet_id': 1},
            'center': [100, 100],
            'target_coordinates': {'px': 100.0, 'py': 200.0, 'pz': 1.5}
        },
        {
            'region_info': {'region_id': 2, 'pallet_id': 1},  
            'center': [200, 100],
            'target_coordinates': {'px': 150.0, 'py': 250.0, 'pz': 2.0}
        },
        {
            'region_info': {'region_id': 3, 'pallet_id': 1},
            'center': [300, 100], 
            'target_coordinates': {'px': 200.0, 'py': 300.0, 'pz': 1.8}
        }
    ]
    
    print("\n1️⃣ Adding pallet to queue...")
    sequencer.add_pallet_to_queue(mock_regions, pallet_id=1)
    
    print("\n2️⃣ Sending regions in sequence [1, 3, 2]...")
    
    # Sequence: P1R1 → P1R3 → P1R2
    for step in [1, 2, 3]:
        print(f"\n--- STEP {step} ---")
        
        # Get next region (triggers callbacks)
        region = sequencer.get_next_region()
        if region:
            region_info = region.get('region_info', {})
            region_id = region_info.get('region_id')
            print(f"🔄 Processing P1R{region_id}...")
            
            # Simulate processing time
            time.sleep(1)
            
            # Robot completed (triggers callbacks)
            print(f"🤖 Robot completed P1R{region_id}")
            sequencer.mark_region_completed()
        else:
            print("❌ No more regions!")
            break
    
    print(f"\n📊 Final Status: {sequencer.status}")

# ===========================================================================
# 🎮 KEYBOARD CONTROL SYSTEM - CÁCH HIỆN TẠI ĐANG DÙNG
# ===========================================================================

def demo_keyboard_triggers():
    """Demo keyboard triggering như trong use_tensorrt_example.py"""
    print("\n🎮 KEYBOARD CONTROL TRIGGERS (Current Implementation)")
    print("=" * 60)
    
    print("Hiện tại system sử dụng keyboard controls để trigger:")
    print("  'n' → get_next_region()      # Trigger IDLE/WAITING → SENDING")
    print("  'c' → mark_region_completed() # Trigger SENDING → WAITING/COMPLETED")
    print("  's' → get_queue_status()     # Show current status")
    print("  'x' → reset_sequence()       # Trigger ANY → IDLE")
    print("  'z' → show_depth_info()      # Additional info")
    
    print("\n💡 Đây chính là cách bạn 'trigger' status hiện tại!")
    print("   Keyboard input → Function call → Status change")

# ===========================================================================
# 🔧 AUTOMATIC TRIGGERING SYSTEM  
# ===========================================================================

class AutomaticTriggerSystem:
    """
    System tự động trigger dựa trên external events.
    """
    
    def __init__(self, sequencer):
        self.sequencer = sequencer
        self.auto_mode = False
        self.robot_completion_signal = False
    
    def monitor_robot_completion(self):
        """Monitor robot completion signal"""
        # Trong thực tế, đây có thể là:
        # - PLC signal
        # - Socket message  
        # - File watcher
        # - Database change
        
        while True:
            if self.robot_completion_signal:
                print("🤖 [AUTO] Robot completion detected!")
                self.sequencer.mark_region_completed()
                self.robot_completion_signal = False
            
            time.sleep(0.1)  # Polling interval
    
    def simulate_robot_signal(self):
        """Simulate robot completion (for demo)"""
        self.robot_completion_signal = True
    
    def enable_auto_sequence(self):
        """Enable automatic sequencing"""
        self.auto_mode = True
        print("🤖 [AUTO] Automatic sequencing enabled")
        
        # Auto send next region when status becomes WAITING
        def auto_send_callback(old_status, new_status):
            if new_status == "WAITING" and self.auto_mode:
                print("🤖 [AUTO] Auto-sending next region...")
                time.sleep(2)  # Simulate robot processing time
                self.simulate_robot_signal()
        
        self.sequencer.on_status_change = auto_send_callback

def demo_automatic_triggers():
    """Demo automatic trigger system"""
    print("\n🤖 AUTOMATIC TRIGGER SYSTEM")
    print("=" * 60)
    
    sequencer = EnhancedRegionSequencer(
        sequence=[1, 3, 2],
        on_status_change=on_status_changed
    )
    
    auto_system = AutomaticTriggerSystem(sequencer)
    auto_system.enable_auto_sequence()
    
    print("✅ Automatic trigger system setup completed!")
    print("💡 Trong thực tế, robot completion signals có thể đến từ:")
    print("   - PLC communication")
    print("   - TCP/UDP socket")
    print("   - File system events")
    print("   - Database triggers")
    print("   - HTTP webhooks")

# ===========================================================================
# 🏃 MAIN DEMO RUNNER
# ===========================================================================

if __name__ == "__main__":
    print("🚀 REGION SEQUENCER STATUS TRIGGERING GUIDE")
    print("=" * 70)
    
    # Demo 1: Callback system
    demo_callback_system()
    
    # Demo 2: Keyboard controls (current method)
    demo_keyboard_triggers()
    
    # Demo 3: Automatic triggers
    demo_automatic_triggers()
    
    print("\n" + "=" * 70)
    print("📝 SUMMARY - CÁCH TRIGGER STATUS:")
    print("1️⃣ CALLBACK SYSTEM: on_status_change() callbacks")
    print("2️⃣ KEYBOARD CONTROL: 'n', 'c', 's', 'x', 'z' keys")  
    print("3️⃣ AUTOMATIC: Robot signals, PLC communication")
    print("4️⃣ MANUAL: Direct function calls")
    print("\n💡 Hiện tại bạn đang dùng method 2️⃣ (keyboard controls)")
    print("   Để enhance, có thể add callback system (method 1️⃣)") 