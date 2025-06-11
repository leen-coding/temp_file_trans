import math
import numpy as np

new_pos = np.array([12,1])
current_pos = np.array([0,22])

tan_theta = (new_pos[1] - current_pos[1]) / (new_pos[0] - current_pos[0])


B_H_total = 2
# B_H_x + B_H_y = B_H_total
# B_H_y = B_H_x * tan_theta
# B_H_y = (B_H_total - B_H_y) * tan_theta
B_H_y = (B_H_total * tan_theta) / (1 + tan_theta)
B_H_x = B_H_total - B_H_y


B_M_total = 1.5
# Bx + By = B_total


# degree = 30
# re1 = math.tan(degree/180*math.pi)

# re2 = math.tan(math.pi - degree/180*math.pi)

# By = (B_total - By) * math.tan(math.pi - degree/180*math.pi)
# By = B_total* math.tan(math.pi - degree/180*math.pi) - By* math.tan(math.pi - degree/180*math.pi)
B_M_y  =   B_M_total* math.tan(math.pi - tan_theta) / (1 + math.tan(math.pi -tan_theta))

B_M_x = B_M_total - B_M_y

print(f'{B_M_x},{B_H_x},{B_M_y},{B_H_y}')


import math

def adjust_vector_to_magnitude_constraints(vector, target_sum):
    """
    调整向量分量，使得其绝对值的和等于给定的 target_sum。
    vector 是一个方向单位向量 (x, y)，我们根据目标绝对值和 target_sum 来缩放它。
    """
    x, y = vector
    abs_sum = abs(x) + abs(y)
    
    if abs_sum == 0:
        return (0, 0)
    
    # 通过缩放，确保 |x| + |y| = target_sum
    scale_factor = target_sum / abs_sum
    return (x * scale_factor, y * scale_factor)

def calculate_direction_vector(pos1, pos2):
    """计算从pos1到pos2的方向向量，并返回单位向量"""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return (0, 0)  # 如果两个点重合，返回零向量
    
    return (dx / magnitude, dy / magnitude)

def dot_product(v1, v2):
    """计算两个向量的点积"""
    return v1[0] * v2[0] + v1[1] * v2[1]

def calculate_fields(current_pos, target_pos, uniform_field_direction, uniform_field_constraint=2, gradient_field_constraint=3):
    """
    计算梯度场方向
    current_pos: 当前坐标
    target_pos: 目标坐标
    uniform_field_direction: 给定的均匀场方向
    uniform_field_constraint: 均匀场的绝对值和约束
    gradient_field_constraint: 梯度场的绝对值和约束
    """
    # 计算当前运动方向向量（标准化为单位向量）
    current_direction = calculate_direction_vector(current_pos, target_pos)
    
    # 调整均匀场，使得其绝对值和为 uniform_field_constraint
    uniform_field = adjust_vector_to_magnitude_constraints(current_direction, uniform_field_constraint)
    
    # 判断当前方向与均匀场的关系，点积为负则表示相反方向
    if dot_product(current_direction, uniform_field_direction) < 0:
        # 反向梯度场
        gradient_field = (-uniform_field[0], -uniform_field[1])
    else:
        # 梯度场与均匀场同向
        gradient_field = uniform_field
    
    # 调整梯度场，使其绝对值和为 gradient_field_constraint
    gradient_field = adjust_vector_to_magnitude_constraints(gradient_field, gradient_field_constraint)
    
    return uniform_field, gradient_field

# 示例使用
current_pos = (1, 1)
target_pos = (12, 1)

# 均匀场的方向假设为沿着 x 轴正方向的 (1, 0)
uniform_field_direction = (1, 0)

# 计算符合约束条件的均匀场和梯度场
uniform_field, gradient_field = calculate_fields(current_pos, target_pos, uniform_field_direction)

print(f"均匀场方向: {uniform_field}")
print(f"梯度场方向: {gradient_field}")
