import math

#----------------------#
# 1. 给定参数（单位） #
#----------------------#
n = 4               # 螺旋匝数
delta_um = 20.0     # 这里假设代表螺旋主半径 σ (µm)
theta_deg = 70.0    # 螺距角 (度)
eta = 1.005e-3      # 水的动力黏度，Pa·s (SI)
r_um = 5.0          # 螺旋丝截面半径 r (µm)
f = 3             # 旋转频率 (Hz)

#--------------------------------------------------------------#
# 2. 单位转换：把 µm 转成 m；把角度(度) 转成弧度(rad)         #
#--------------------------------------------------------------#
delta_m = delta_um * 1e-6
r_m     = r_um     * 1e-6
theta_rad = math.radians(theta_deg)

#--------------------------------------------------------------#
# 3. 计算平动拖曳系数 ζ⊥ 和 ζ∥ (Lighthill公式 11,12)          #
#    注意 log(...) 的自变量必须无量纲；下面用 σ/(r·sinθ)。     #
#--------------------------------------------------------------#
zeta_perp = (4.0 * math.pi * eta) / (
    math.log((0.36 * math.pi * delta_m) / (r_m * math.sin(theta_rad))) + 0.5
)
zeta_para = (2.0 * math.pi * eta) / (
    math.log((0.36 * math.pi * delta_m) / (r_m * math.sin(theta_rad)))
)

#--------------------------------------------------------------#
# 4. 计算推进矩阵中的 a,b,c (参见文献式(8)-(10))              #
#   若 σ=delta_m, 则:                                         #
#   a = 2πnσ * [ (ζ∥cos²θ + ζ⊥sin²θ) / sinθ ]                   #
#   b = 2πnσ² (ζ∥-ζ⊥)cosθ                                      #
#   c = 2πnσ³ * [ (ζ⊥cos²θ + ζ∥sin²θ) / sinθ ]                   #
#--------------------------------------------------------------#
a = 2.0 * math.pi * n * delta_m * (
    (zeta_para * math.cos(theta_rad)**2 + zeta_perp * math.sin(theta_rad)**2)
    / math.sin(theta_rad)
)
b = 2.0 * math.pi * n * (delta_m**2) * (zeta_para - zeta_perp) * math.cos(theta_rad)
c = 2.0 * math.pi * n * (delta_m**3) * (
    (zeta_perp * math.cos(theta_rad)**2 + zeta_para * math.sin(theta_rad)**2)
    / math.sin(theta_rad)
)

#--------------------------------------------------------------#
# 5. 在无外力(f=0)下，螺旋自由旋转的扭矩：                   #
#   τ = [ c - (b² / a) ] * ω,   其中 ω = 2πf                   #
#   注意很多人会误写成 (ac-b)/a，正确应是 (ac - b²)/a         #
#--------------------------------------------------------------#
omega = 2.0 * math.pi * f
torque = (c - (b**2 / a)) * omega

#-----------------------#
# 6. 输出结果(单位检查) #
#-----------------------#
print(f"a = {a:.3e}  (N·s/m)   # 平动阻力量级×半径 => 力/速度 视作推广系数")
print(f"b = {b:.3e}  (N·s)      # 耦合项   (也要留意实际含义)")
print(f"c = {c:.3e}  (N·m·s)    # 旋转阻力系数 => 扭矩/角速度")
print(f"Torque = {torque:.3e} N·m  at f={f} Hz")


def required_ni_thickness(
    torque_n_m: float,    # 目标扭矩(N·m)
    area_um2: float,      # 需要镀镍的表面积(µm^2)
    Ms: float,            # 镍的饱和磁化强度(A/m)
    B: float              # 外磁场强度(T)
) -> float:
    """
    根据 τ = m·B = (Ms·Area·Thickness)·B 解出所需镀镍厚度。
    返回值单位：米(m)。
    """
    # 1. 把表面积从 µm^2 转换为 m^2
    area_m2 = area_um2 * 1e-12  # 1 µm^2 = 1e-12 m^2
    
    # 2. 厚度 = τ / (Ms * B * Area)
    thickness_m = torque_n_m / (Ms * B * area_m2)
    return thickness_m

#-------------------------------------------#
#  以下是一个示例使用                       #
#-------------------------------------------#

if __name__ == "__main__":
    # 已知参数:
    area_um2 = 16098.0          # 表面积 (µm^2)
    Ms = 5.0e5                  # Ni饱和磁化强度 (A/m), ~ (4.9~5.0)e5
    B = 0.001                   # 假设外磁场 5mT = 0.005 T
    
    # 例如你计算的目标扭矩:
    torque_1hz = 5.626e-15      # 1 Hz 时需要的扭矩 (N·m)
    torque_3hz = 1.688e-14      # 3 Hz 时需要的扭矩 (N·m)

    # 分别计算厚度
    thickness_1hz_m = required_ni_thickness(torque_1hz, area_um2, Ms, B)
    thickness_3hz_m = required_ni_thickness(torque_3hz, area_um2, Ms, B)

    # 转成纳米
    thickness_1hz_nm = thickness_1hz_m * 1e9
    thickness_3hz_nm = thickness_3hz_m * 1e9

    print(f"For torque {torque_1hz:.2e} N·m at 1 Hz:")
    print(f"  Required Ni thickness ~ {thickness_1hz_nm:.2f} nm  (B={B*1e3} mT)\n")

    print(f"For torque {torque_3hz:.2e} N·m at 3 Hz:")
    print(f"  Required Ni thickness ~ {thickness_3hz_nm:.2f} nm  (B={B*1e3} mT)")