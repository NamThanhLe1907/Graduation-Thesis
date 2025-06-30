"""
Quick Fix Script cho Bag Control Keys
Tự động set environment variables và test bag control
"""
import os
import subprocess
import sys

def set_environment_variables():
    """Tự động set các environment variables cần thiết"""
    print("🔧 [QUICK FIX] Setting environment variables...")
    
    # Set ENABLE_PLC=true
    os.environ['ENABLE_PLC'] = 'true'
    print("   ✅ Set ENABLE_PLC=true")
    
    # Set PLC_IP if not already set
    if 'PLC_IP' not in os.environ:
        os.environ['PLC_IP'] = '192.168.0.1'
        print("   ✅ Set PLC_IP=192.168.0.1 (default)")
    else:
        print(f"   ℹ️ PLC_IP already set: {os.environ['PLC_IP']}")
    
    # Set other useful variables
    os.environ['WORKER_LOGGING_DISABLED'] = 'false'  # Enable logging for debug
    print("   ✅ Set WORKER_LOGGING_DISABLED=false")

def test_bag_control_simple():
    """Simple test của bag control functionality"""
    print("\n🧪 [SIMPLE TEST] Testing bag control functions...")
    
    try:
        from region_division_plc_integration import RegionDivisionPLCIntegration
        
        # Create PLC integration instance
        plc_integration = RegionDivisionPLCIntegration(
            plc_ip=os.environ.get('PLC_IP', '192.168.0.1'),
            debug=True
        )
        
        print("   ✅ PLC Integration created successfully")
        
        # Test bag 1
        print("\n   🎯 Testing Bag 1:")
        plc_integration.set_current_bag_number(1)
        bag_info = plc_integration.get_current_bag_info()
        print(f"      Result: {bag_info['sequence_mapping']}")
        
        # Test bag 2
        print("\n   🎯 Testing Bag 2:")
        plc_integration.set_current_bag_number(2)
        bag_info = plc_integration.get_current_bag_info()
        print(f"      Result: {bag_info['sequence_mapping']}")
        
        # Test bag 3
        print("\n   🎯 Testing Bag 3:")
        plc_integration.set_current_bag_number(3)
        bag_info = plc_integration.get_current_bag_info()
        print(f"      Result: {bag_info['sequence_mapping']}")
        
        print("\n   ✅ All bag control functions work correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Bag control test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_camera_standalone_with_debug():
    """Chạy camera_standalone.py với debug enabled"""
    print("\n🚀 [LAUNCH] Starting camera_standalone.py with debug...")
    
    try:
        # Import and run camera_standalone
        from camera_standalone import demo_camera
        
        print("   📋 Environment variables set:")
        print(f"      ENABLE_PLC = {os.environ.get('ENABLE_PLC')}")
        print(f"      PLC_IP = {os.environ.get('PLC_IP')}")
        
        print("\n   🎯 Starting camera demo...")
        print("   💡 After pipeline starts, try pressing keys 1, 2, 3")
        print("   💡 Look for debug messages in console")
        
        demo_camera()
        
    except KeyboardInterrupt:
        print("\n   ⚠️ Camera demo interrupted by user")
    except Exception as e:
        print(f"\n   ❌ Camera demo failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🚀 QUICK FIX BAG CONTROL SCRIPT")
    print("="*50)
    
    # Step 1: Set environment variables
    set_environment_variables()
    
    # Step 2: Test bag control functions
    bag_control_ok = test_bag_control_simple()
    
    if not bag_control_ok:
        print("\n❌ [ERROR] Bag control functions don't work!")
        print("   Cannot proceed with camera demo")
        return
    
    # Step 3: Ask user if they want to run camera demo
    print("\n🤔 [CHOICE] Do you want to run camera_standalone.py now?")
    choice = input("   Type 'y' or 'yes' to run, any other key to exit: ").strip().lower()
    
    if choice in ['y', 'yes']:
        run_camera_standalone_with_debug()
    else:
        print("\n📋 [MANUAL STEPS] To run manually:")
        print("   1. Make sure these environment variables are set:")
        print(f"      set ENABLE_PLC=true")
        print(f"      set PLC_IP={os.environ.get('PLC_IP', '192.168.0.1')}")
        print("   2. Run: python camera_standalone.py")
        print("   3. Wait for pipeline to start")
        print("   4. Press keys 1, 2, 3 to test bag control")

if __name__ == "__main__":
    main() 