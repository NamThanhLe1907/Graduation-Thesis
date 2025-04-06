import tkinter as tk
from threading import Thread
from utility.visualization import AppGUI
from utility.video_processor import VideoProcessor


def main():
    root = tk.Tk()
    gui = AppGUI(root)
    processor = VideoProcessor(gui)

    processing_thread = Thread(target=processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()

    print("Ứng dụng đang khởi động...")
    gui.run()
    print("Ứng dụng đã tắt")

if __name__ == "__main__":
    print("Bắt đầu chạy chương trình từ main")
    main()
