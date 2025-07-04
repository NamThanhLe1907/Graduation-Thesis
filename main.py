import tkinter as tk
from threading import Thread
from utility.archived.visualization import AppGUI
from utility.archived.video_processor_v2 import VideoProcessorV2


def main():
    root = tk.Tk()
    gui = AppGUI(root)
    processor = VideoProcessorV2(gui)

    processing_thread = Thread(target=processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()

    print("Ứng dụng đang khởi động...")
    gui.run()
    print("Ứng dụng đã tắt")

if __name__ == "__main__":
    print("Bắt đầu chạy chương trình từ main")
    main()
