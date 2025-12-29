import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog

# ==================== 配置参数 ====================
# 输入目录：包含.ulg文件的目录（结果将保存在该目录下的Result文件夹）
INPUT_DIR = "ULOG_3"  # 可以修改为其他路径，例如: "F:\\logs" 或 "ULOG"

# 电流系数：用于修正电池电流数据的偏差（默认为1.0，表示不修正）
CURRENT_COEFFICIENT = 1  # 根据实际情况调整此系数

# 设置要绘制的遥控器通道编号（1-18），可以是单个或多个通道
RC_CHANNELS_TO_PLOT = [1, 2, 3, 4]
# ==================================================

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def quaternion_to_euler(q0, q1, q2, q3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quaternions to Euler angles (roll, pitch, yaw) in degrees."""
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    q3 = np.asarray(q3)

    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def extract_position_velocity_acceleration(ulog: ULog) -> pd.DataFrame:
    """Extract position, velocity, and acceleration from vehicle_local_position topic."""
    dataset = next((d for d in ulog.data_list if d.name == "vehicle_local_position"), None)
    if dataset is None:
        raise RuntimeError("Topic 'vehicle_local_position' not found in log")

    data = dataset.data
    ts = np.asarray(data["timestamp"], dtype=np.float64)
    time_s = (ts - ts[0]) * 1e-6

    pos_vel_acc_df = pd.DataFrame({
        "time_s": time_s,
        # 位置 (NED坐标系)
        "pos_x_m": data["x"],
        "pos_y_m": data["y"],
        "pos_z_m": data["z"],
        # 速度 (NED坐标系)
        "vel_x_ms": data["vx"],
        "vel_y_ms": data["vy"],
        "vel_z_ms": data["vz"],
        # 加速度 (如果有的话)
        "acc_x_ms2": data.get("ax", np.zeros_like(time_s)),
        "acc_y_ms2": data.get("ay", np.zeros_like(time_s)),
        "acc_z_ms2": data.get("az", np.zeros_like(time_s)),
    })
    return pos_vel_acc_df


def extract_attitude_and_rates(ulog: ULog) -> pd.DataFrame:
    """Extract attitude (Euler angles), angular velocity, and angular acceleration."""
    # 提取姿态角
    attitude_dataset = next((d for d in ulog.data_list if d.name == "vehicle_attitude"), None)
    if attitude_dataset is None:
        raise RuntimeError("Topic 'vehicle_attitude' not found in log")

    att_data = attitude_dataset.data
    ts = np.asarray(att_data["timestamp"], dtype=np.float64)
    time_s = (ts - ts[0]) * 1e-6

    roll_deg, pitch_deg, yaw_deg = quaternion_to_euler(
        att_data["q[0]"], att_data["q[1]"], att_data["q[2]"], att_data["q[3]"]
    )

    # 提取角速度
    rates_dataset = next((d for d in ulog.data_list if d.name == "vehicle_angular_velocity"), None)
    if rates_dataset:
        rates_data = rates_dataset.data
        rates_ts = np.asarray(rates_data["timestamp"], dtype=np.float64)
        rates_time_s = (rates_ts - rates_ts[0]) * 1e-6
        
        # 插值到统一时间轴
        roll_rate = np.interp(time_s, rates_time_s, rates_data["xyz[0]"])
        pitch_rate = np.interp(time_s, rates_time_s, rates_data["xyz[1]"])
        yaw_rate = np.interp(time_s, rates_time_s, rates_data["xyz[2]"])
    else:
        roll_rate = np.zeros_like(time_s)
        pitch_rate = np.zeros_like(time_s)
        yaw_rate = np.zeros_like(time_s)

    # 计算角加速度（通过数值微分）
    dt = np.diff(time_s)
    dt = np.append(dt, dt[-1])  # 保持长度一致
    
    roll_acc = np.gradient(np.degrees(roll_rate), time_s)
    pitch_acc = np.gradient(np.degrees(pitch_rate), time_s)
    yaw_acc = np.gradient(np.degrees(yaw_rate), time_s)

    attitude_df = pd.DataFrame({
        "time_s": time_s,
        # 姿态角 (度)
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        # 角速度 (度/秒)
        "roll_rate_degs": np.degrees(roll_rate),
        "pitch_rate_degs": np.degrees(pitch_rate),
        "yaw_rate_degs": np.degrees(yaw_rate),
        # 角加速度 (度/秒²)
        "roll_acc_degs2": roll_acc,
        "pitch_acc_degs2": pitch_acc,
        "yaw_acc_degs2": yaw_acc,
    })
    return attitude_df


def extract_rc_channels(ulog: ULog) -> pd.DataFrame:
    """Extract RC (remote control) channel values."""
    dataset = next((d for d in ulog.data_list if d.name == "input_rc"), None)
    if dataset is None:
        raise RuntimeError("Topic 'input_rc' not found in log")

    data = dataset.data
    ts = np.asarray(data["timestamp"], dtype=np.float64)
    time_s = (ts - ts[0]) * 1e-6

    rc_df = pd.DataFrame({"time_s": time_s})
    
    for key in data.keys():
        if key.startswith("values[") and key.endswith("]"):
            rc_df[key.replace("values", "channel")] = data[key]
    
    return rc_df


def extract_battery(ulog: ULog) -> pd.DataFrame:
    """Extract battery voltage and current."""
    dataset = next((d for d in ulog.data_list if d.name == "battery_status"), None)
    if dataset is None:
        raise RuntimeError("Topic 'battery_status' not found in log")

    data = dataset.data
    ts = np.asarray(data["timestamp"], dtype=np.float64)
    time_s = (ts - ts[0]) * 1e-6

    # 应用电流系数进行修正
    corrected_current = np.asarray(data["current_a"]) * CURRENT_COEFFICIENT

    battery_df = pd.DataFrame({
        "time_s": time_s,
        "voltage_v": data["voltage_v"],
        "current_a": corrected_current,
    })
    return battery_df


def plot_comprehensive(pos_vel_acc_df: pd.DataFrame, attitude_df: pd.DataFrame, 
                      rc_df: pd.DataFrame, battery_df: pd.DataFrame, 
                      output_path: Path) -> None:
    """Plot comprehensive flight dynamics data."""
    fig = plt.figure(figsize=(16, 15))
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)
    
    # 第一行：位置 (3个子图)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["pos_x_m"], 'b-', linewidth=1.5)
    ax1.set_ylabel("X Position [m]", fontsize=10)
    ax1.set_title("Position (NED)", fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["pos_y_m"], 'g-', linewidth=1.5)
    ax2.set_ylabel("Y Position [m]", fontsize=10)
    ax2.set_title("Position (NED)", fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["pos_z_m"], 'r-', linewidth=1.5)
    ax3.set_ylabel("Z Position [m]", fontsize=10)
    ax3.set_title("Position (NED)", fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    # 第二行：速度 (3个子图)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["vel_x_ms"], 'b-', linewidth=1.5)
    ax4.set_ylabel("Vx [m/s]", fontsize=10)
    ax4.set_title("Velocity", fontsize=11, fontweight='bold')
    ax4.grid(True, linestyle="--", alpha=0.5)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["vel_y_ms"], 'g-', linewidth=1.5)
    ax5.set_ylabel("Vy [m/s]", fontsize=10)
    ax5.set_title("Velocity", fontsize=11, fontweight='bold')
    ax5.grid(True, linestyle="--", alpha=0.5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["vel_z_ms"], 'r-', linewidth=1.5)
    ax6.set_ylabel("Vz [m/s]", fontsize=10)
    ax6.set_title("Velocity", fontsize=11, fontweight='bold')
    ax6.grid(True, linestyle="--", alpha=0.5)
    
    # 第三行：加速度 (3个子图)
    ax6a = fig.add_subplot(gs[2, 0])
    ax6a.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["acc_x_ms2"], 'b-', linewidth=1.5)
    ax6a.set_ylabel("Ax [m/s²]", fontsize=10)
    ax6a.set_title("Acceleration", fontsize=11, fontweight='bold')
    ax6a.grid(True, linestyle="--", alpha=0.5)
    
    ax6b = fig.add_subplot(gs[2, 1])
    ax6b.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["acc_y_ms2"], 'g-', linewidth=1.5)
    ax6b.set_ylabel("Ay [m/s²]", fontsize=10)
    ax6b.set_title("Acceleration", fontsize=11, fontweight='bold')
    ax6b.grid(True, linestyle="--", alpha=0.5)
    
    ax6c = fig.add_subplot(gs[2, 2])
    ax6c.plot(pos_vel_acc_df["time_s"], pos_vel_acc_df["acc_z_ms2"], 'r-', linewidth=1.5)
    ax6c.set_ylabel("Az [m/s²]", fontsize=10)
    ax6c.set_title("Acceleration", fontsize=11, fontweight='bold')
    ax6c.grid(True, linestyle="--", alpha=0.5)
    
    # 第四行：姿态角、角速度、角加速度
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(attitude_df["time_s"], attitude_df["roll_deg"], 'b-', label="Roll", linewidth=1.5)
    ax7.plot(attitude_df["time_s"], attitude_df["pitch_deg"], 'g-', label="Pitch", linewidth=1.5)
    ax7.plot(attitude_df["time_s"], attitude_df["yaw_deg"], 'r-', label="Yaw", linewidth=1.5)
    ax7.set_ylabel("Angle [deg]", fontsize=10)
    ax7.set_title("Attitude", fontsize=11, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, linestyle="--", alpha=0.5)
    
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(attitude_df["time_s"], attitude_df["roll_rate_degs"], 'b-', label="Roll", linewidth=1.5)
    ax8.plot(attitude_df["time_s"], attitude_df["pitch_rate_degs"], 'g-', label="Pitch", linewidth=1.5)
    ax8.plot(attitude_df["time_s"], attitude_df["yaw_rate_degs"], 'r-', label="Yaw", linewidth=1.5)
    ax8.set_ylabel("Rate [deg/s]", fontsize=10)
    ax8.set_title("Angular Velocity", fontsize=11, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.grid(True, linestyle="--", alpha=0.5)
    
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.plot(attitude_df["time_s"], attitude_df["roll_acc_degs2"], 'b-', label="Roll", linewidth=1.5)
    ax9.plot(attitude_df["time_s"], attitude_df["pitch_acc_degs2"], 'g-', label="Pitch", linewidth=1.5)
    ax9.plot(attitude_df["time_s"], attitude_df["yaw_acc_degs2"], 'r-', label="Yaw", linewidth=1.5)
    ax9.set_ylabel("Acc [deg/s²]", fontsize=10)
    ax9.set_title("Angular Acceleration", fontsize=11, fontweight='bold')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.grid(True, linestyle="--", alpha=0.5)
    
    # 第五行：遥控器和电池 (跨列)
    ax10 = fig.add_subplot(gs[4, 0:2])
    for ch_num in RC_CHANNELS_TO_PLOT:
        col_name = f"channel[{ch_num-1}]"
        if col_name in rc_df.columns:
            ax10.plot(rc_df["time_s"], rc_df[col_name], label=f"CH{ch_num}", linewidth=1.5)
    ax10.set_xlabel("Time [s]", fontsize=10)
    ax10.set_ylabel("Channel Value", fontsize=10)
    ax10.set_title("RC Channels", fontsize=11, fontweight='bold')
    ax10.legend(loc='upper right', fontsize=8)
    ax10.grid(True, linestyle="--", alpha=0.5)
    
    ax11 = fig.add_subplot(gs[4, 2])
    ax11_twin = ax11.twinx()
    line1 = ax11.plot(battery_df["time_s"], battery_df["voltage_v"], 'b-', label="Voltage", linewidth=1.5)
    line2 = ax11_twin.plot(battery_df["time_s"], battery_df["current_a"], 'r-', label="Current", linewidth=1.5)
    ax11.set_xlabel("Time [s]", fontsize=10)
    ax11.set_ylabel("Voltage [V]", fontsize=10, color='b')
    ax11_twin.set_ylabel("Current [A]", fontsize=10, color='r')
    ax11.set_title("Battery", fontsize=11, fontweight='bold')
    ax11.tick_params(axis='y', labelcolor='b')
    ax11_twin.tick_params(axis='y', labelcolor='r')
    ax11.grid(True, linestyle="--", alpha=0.5)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax11.legend(lines, labels, loc='upper right', fontsize=8)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def process_single_log(log_path: Path, output_base_dir: Path = None) -> bool:
    """
    Process a single ULog file.
    
    Args:
        log_path: Path to the .ulg file
        output_base_dir: Base directory for outputs. If None, creates folder next to log file.
    
    Returns:
        True if processed, False if skipped
    """
    if output_base_dir is None:
        output_dir = log_path.with_suffix("")
    else:
        output_dir = output_base_dir / log_path.stem
    
    # 检查是否已经处理过（检查关键文件是否存在）
    plot_path = output_dir / "dynamics_analysis.png"
    pos_vel_acc_path = output_dir / "位置速度加速度.xlsx"
    attitude_path = output_dir / "姿态角速度加速度.xlsx"
    rc_excel_path = output_dir / "遥控器.xlsx"
    battery_excel_path = output_dir / "电池.xlsx"
    
    if (plot_path.exists() and pos_vel_acc_path.exists() and 
        attitude_path.exists() and rc_excel_path.exists() and battery_excel_path.exists()):
        print(f"⏭️  跳过 (已处理): {log_path.name}")
        return False
    
    print(f"🔄 处理中: {log_path.name}")
    
    try:
        # 加载 ULog
        ulog = ULog(str(log_path))
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取数据
        pos_vel_acc_df = extract_position_velocity_acceleration(ulog)
        attitude_df = extract_attitude_and_rates(ulog)
        rc_df = extract_rc_channels(ulog)
        battery_df = extract_battery(ulog)
        
        # 保存Excel文件
        pos_vel_acc_df.to_excel(pos_vel_acc_path, index=False)
        attitude_df.to_excel(attitude_path, index=False)
        rc_df.to_excel(rc_excel_path, index=False)
        battery_df.to_excel(battery_excel_path, index=False)
        
        # 绘制图像
        plot_comprehensive(pos_vel_acc_df, attitude_df, rc_df, battery_df, plot_path)
        
        print(f"✅ 完成: {log_path.name} -> {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 错误: {log_path.name} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_process(input_dir: Path, output_base_dir: Path = None, recursive: bool = False) -> None:
    """
    Batch process all .ulg files in a directory.
    
    Args:
        input_dir: Directory containing .ulg files
        output_base_dir: Base directory for outputs. If None, creates folders next to log files.
        recursive: If True, search for .ulg files recursively in subdirectories
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有.ulg文件
    pattern = "**/*.ulg" if recursive else "*.ulg"
    ulg_files = sorted(input_dir.glob(pattern))
    
    if not ulg_files:
        print(f"⚠️  未找到.ulg文件: {input_dir}")
        return
    
    print(f"\n📁 扫描目录: {input_dir}")
    print(f"📊 找到 {len(ulg_files)} 个.ulg文件\n")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, log_path in enumerate(ulg_files, 1):
        print(f"[{i}/{len(ulg_files)}] ", end="")
        result = process_single_log(log_path, output_base_dir)
        
        if result:
            processed_count += 1
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"📈 处理统计:")
    print(f"   ✅ 已处理: {processed_count}")
    print(f"   ⏭️  已跳过: {skipped_count}")
    print(f"   ❌ 错误: {error_count}")
    print(f"   📊 总计: {len(ulg_files)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description=f"批量处理PX4 ULog文件，分析飞行动力学数据（位置、速度、加速度、姿态等）（当前输入目录: {INPUT_DIR}，结果保存在 {INPUT_DIR}/Result）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理配置的输入目录（默认ULOG_2）
  python batch_parse_px4_ulog_dynamics.py
  
  # 递归处理所有子目录中的.ulg文件
  python batch_parse_px4_ulog_dynamics.py --recursive
  
注意: 要修改输入目录，请直接编辑脚本开头的 INPUT_DIR 参数
        """
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归搜索子目录中的.ulg文件"
    )

    args = parser.parse_args()
    
    # 使用配置的输入目录
    input_dir = Path.cwd() / INPUT_DIR if not Path(INPUT_DIR).is_absolute() else Path(INPUT_DIR)
    output_dir = input_dir / "Result"
    
    try:
        batch_process(input_dir, output_dir, args.recursive)
        print("🎉 全部处理完成！")
    except Exception as e:
        print(f"\n❌ 致命错误: {e}")
        print("处理过程中出现错误，程序终止。")
        raise


if __name__ == "__main__":
    main()
