import cv2

class Visualizer:
    def __init__(self):
        pass

    def draw_boxes(self, frame, boxes, confidences, class_ids, class_names, show_metadata=True):
        """Draw bounding boxes and metadata on the frame."""
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if show_metadata:
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def show_frame(self, frame):
        """Display the frame in a window."""
        cv2.imshow("Detection", frame)
        cv2.waitKey(1)  # Display for 1 ms

if __name__ == "__main__":
    visualizer = Visualizer()
    # Example usage
    frame = cv2.imread('sample_image.jpg')  # Replace with actual frame capture
    boxes = [[100, 100, 200, 200]]  # Example bounding box
    confidences = [0.95]  # Example confidence
    class_ids = [0]  # Example class ID
    class_names = ["pallet", "load"]  # Example class names
    frame_with_boxes = visualizer.draw_boxes(frame, boxes, confidences, class_ids, class_names)
    visualizer.show_frame(frame_with_boxes)
    cv2.destroyAllWindows()