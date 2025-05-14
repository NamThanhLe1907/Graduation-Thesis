import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageOps
import cv2
import time

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Processor")
        self.root.geometry("1300x800")
        
        # Sử dụng PanedWindow để chia giao diện thành 2 phần: Annotated và Depth Map
        self.paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame bên trái: Hiển thị Annotated Frame
        self.annotated_frame = ttk.Frame(self.paned, width=650, relief=tk.SUNKEN)
        self.paned.add(self.annotated_frame, weight=1)
        
        # Frame bên phải: Hiển thị Depth Map (Heatmap)
        self.depth_frame = ttk.Frame(self.paned, width=650, relief=tk.SUNKEN)
        self.paned.add(self.depth_frame, weight=1)
        
        # Label hiển thị Annotated Image
        self.annotated_label = tk.Label(self.annotated_frame)
        self.annotated_label.pack(side=tk.TOP, padx=5, pady=5, expand=True)
        
        # Label hiển thị Depth Map
        self.depth_label = tk.Label(self.depth_frame)
        self.depth_label.pack(side=tk.TOP, padx=5, pady=5, expand=True)
        
        # Khung thông tin bên dưới, chứa FPS, Log và nút Exit
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.fps_label = tk.Label(self.info_frame, text="FPS: 0.0", font=("Helvetica", 12))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.console = tk.Text(self.info_frame, height=5, width=80)
        self.console.pack(side=tk.LEFT, padx=10)
        
        self.exit_btn = tk.Button(self.info_frame, text="Exit", command=self.root.destroy,
                                  bg="#ff4444", fg="white", font=("Helvetica", 12))
        self.exit_btn.pack(side=tk.RIGHT, padx=10)
        
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
        self.annotated_label.image = annotated_imgtk  # Giữ tham chiếu
        self.annotated_label.update_idletasks()
        self.annotated_label.update()
        
    def update_depth(self, depth_map):
        """
        Cập nhật giao diện với depth map (heatmap).
        """
        try:
            # Chuyển từ BGR sang RGB để hiển thị đúng màu sắc
            depth_rgb = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.log_message(f"Error converting depth map: {e}", "ERROR")
            return
        
        depth_img = Image.fromarray(depth_rgb)
        # Thêm viền đỏ để phân biệt heatmap (tuỳ chọn)
        depth_img = ImageOps.expand(depth_img, border=5, fill='red')
        depth_imgtk = ImageTk.PhotoImage(image=depth_img)
        
        self.depth_label.config(image=depth_imgtk)
        self.depth_label.image = depth_imgtk  # Giữ tham chiếu
        self.depth_label.update_idletasks()
        self.depth_label.update()
        
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
