import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Dùng cho đồ họa 3D

# Định nghĩa 2 vector 3D
v1 = np.array([3, 2, 1])
v2 = np.array([2, 3, 4])

# Tính dot product
dot = np.dot(v1, v2)

# Tính phép chiếu của v2 lên v1: proj_v2_on_v1 = (v2 · v1 / ||v1||²) * v1
proj_v2_on_v1 = (np.dot(v2, v1) / np.dot(v1, v1)) * v1

# Thành phần vuông góc: v2_perp = v2 - proj_v2_on_v1
v2_perp = v2 - proj_v2_on_v1

# Tạo đồ họa 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
origin = [0, 0, 0]

# Vẽ vector v1 (màu đỏ)
ax.quiver(origin[0], origin[1], origin[2],
          v1[0], v1[1], v1[2], color='r', label='v1', arrow_length_ratio=0.1)

# Vẽ vector v2 (màu xanh)
ax.quiver(origin[0], origin[1], origin[2],
          v2[0], v2[1], v2[2], color='b', label='v2', arrow_length_ratio=0.1)

# Vẽ phép chiếu của v2 lên v1 (màu xanh lá)
ax.quiver(origin[0], origin[1], origin[2],
          proj_v2_on_v1[0], proj_v2_on_v1[1], proj_v2_on_v1[2],
          color='g', label='Proj của v2 lên v1', arrow_length_ratio=0.1)

# Vẽ thành phần vuông góc từ đỉnh của proj đến đỉnh của v2 (màu tím, kiểu dashed)
ax.quiver(proj_v2_on_v1[0], proj_v2_on_v1[1], proj_v2_on_v1[2],
          v2_perp[0], v2_perp[1], v2_perp[2],
          color='m', label='Phần vuông góc', arrow_length_ratio=0.1, linestyle='dashed')

# Thiết lập giới hạn cho các trục (dựa trên độ dài vector lớn nhất)
max_range = max(np.linalg.norm(v1), np.linalg.norm(v2)) + 1
ax.set_xlim([0, max_range])
ax.set_ylim([0, max_range])
ax.set_zlim([0, max_range])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f"3D Dot Product Visualization\nDot = {dot:.2f}")
ax.legend()

plt.show()
