import cv2
import numpy as np

class Calibration:
    def __init__(self, image_points, robot_points):
        self.image_points = np.array(image_points, dtype=np.float32)
        self.robot_points = np.array(robot_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(
            self.image_points, 
            self.robot_points
        )
        if self.homography_matrix is None:
            raise ValueError("Không thể tính Homography Matrix")


    @staticmethod
    def pixel_to_robot(pixel_coord, homography_matrix):
        pixel_coord_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
        robot_coord_homogeneous = np.dot(homography_matrix, pixel_coord_homogeneous)
        robot_coord_homogeneous /= robot_coord_homogeneous[2]
        return robot_coord_homogeneous[:2]


# # Ví dụ sử dụng
# if __name__ == "__main__":
#     image_points = [[1118, 287], [1118, 709], [635, 709], [635, 287]]
#     robot_points = [[-80, 120], [-80, 308], [105, 308], [105, 120]]
#     calibration = Calibration(image_points, robot_points)

#     pixel_coord = [800, 500]
#     robot_coord = calibration.pixel_to_robot(pixel_coord)
#     print("Tọa độ robot tương ứng:", robot_coord)