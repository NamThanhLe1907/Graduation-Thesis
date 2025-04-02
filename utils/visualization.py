import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Processor")
        self.root.geometry("1300x800")
        
        # Khung hiển thị hình annotated với chiều cao cố định
        self.image_frame = tk.Frame(root, height=600)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.image_frame.pack_propagate(False)
        
        self.annotated_label = tk.Label(self.image_frame)
        self.annotated_label.pack(side=tk.TOP, padx=5, pady=5, expand=True)
        
        # Khung thông tin: FPS và log message
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.fps_label = tk.Label(self.info_frame, text="FPS: 0.0", font=("Helvetica", 12))
        self.fps_label.pack(pady=5)
        
        self.console = tk.Text(self.info_frame, height=10, width=100)
        self.console.pack(pady=5)
        
    def update(self, annotated):
        """
        Cập nhật giao diện với annotated frame.
        """
        try:
            # Chuyển từ BGR sang RGB để hiển thị đúng màu sắc
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.log_message(f"Error converting annotated frame: {e}", "ERROR")
            return
        
        annotated_img = Image.fromarray(annotated_rgb)
        annotated_imgtk = ImageTk.PhotoImage(image=annotated_img)
        
        self.annotated_label.config(image=annotated_imgtk)
        self.annotated_label.image = annotated_imgtk  # Giữ tham chiếu để tránh bị thu gom rác
        self.annotated_label.update_idletasks()
        self.annotated_label.update()
        
    def log_message(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}\n"
        self.root.after(0, lambda: (self.console.insert(tk.END, entry), self.console.see(tk.END)))
        
    def update_fps_info(self, fps, avg_inference):
        info_text = f"FPS: {fps:.2f} | Avg Inference: {avg_inference:.3f} s"
        self.fps_label.config(text=info_text)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    gui = AppGUI(root)
    # Demo: load hình mẫu từ file nếu có
    frame = cv2.imread("sample_image.jpg")
    if frame is not None:
        gui.update(frame)
    gui.run()
