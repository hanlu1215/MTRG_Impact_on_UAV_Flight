import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog

# ==================== 配置参数 ====================
# 设置要绘制的遥控器通道编号（1-18），可以是单个或多个通道
# 例如: [5] 表示只绘制通道5
#      [1, 2, 3, 4] 表示绘制通道1-4
RC_CHANNELS_TO_PLOT = [6]
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


def extract_attitude(ulog: ULog) -> pd.DataFrame:
    """Extract roll/pitch/yaw from the vehicle_attitude topic."""
    dataset = next((d for d in ulog.data_list if d.name == "vehicle_attitude"), None)
    if dataset is None:
        raise RuntimeError("Topic 'vehicle_attitude' not found in log")

    data = dataset.data
    ts = np.asarray(data["timestamp"], dtype=np.float64)
    time_s = (ts - ts[0]) * 1e-6

    roll_deg, pitch_deg, yaw_deg = quaternion_to_euler(
        data["q[0]"], data["q[1]"], data["q[2]"], data["q[3]"]
    )

    attitude_df = pd.DataFrame(
        {
            "time_s": time_s,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
        }
    )
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

    battery_df = pd.DataFrame(
        {
            "time_s": time_s,
            "voltage_v": data["voltage_v"],
            "current_a": data["current_a"],
        }
    )
    return battery_df


def plot_combined(attitude_df: pd.DataFrame, rc_df: pd.DataFrame, battery_df: pd.DataFrame, output_path: Path) -> None:
    """Plot attitude, RC channels, and battery data in a single figure with 3 subplots."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Attitude angles
    ax1.plot(attitude_df["time_s"], attitude_df["roll_deg"], label="Roll", linewidth=1.5)
    ax1.plot(attitude_df["time_s"], attitude_df["pitch_deg"], label="Pitch", linewidth=1.5)
    ax1.set_title("Attitude Angles", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Angle [deg]", fontsize=10)
    ax1.set_ylim(-20, 20)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Plot 2: RC channels (plot specified channels from RC_CHANNELS_TO_PLOT)
    for ch_num in RC_CHANNELS_TO_PLOT:
        col_name = f"channel[{ch_num-1}]"
        if col_name in rc_df.columns:
            ax2.plot(rc_df["time_s"], rc_df[col_name], label=f"{ch_num}", linewidth=1.5)
    ax2.set_title("RC Channels", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Channel Value", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc='upper right', ncol=min(4, len(RC_CHANNELS_TO_PLOT)))
    
    # Plot 3: Battery voltage and current
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(battery_df["time_s"], battery_df["voltage_v"], 'b-', label="Voltage", linewidth=1.5)
    line2 = ax3_twin.plot(battery_df["time_s"], battery_df["current_a"], 'r-', label="Current", linewidth=1.5)
    
    ax3.set_title("Battery Status", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Time [s]", fontsize=10)
    ax3.set_ylabel("Voltage [V]", fontsize=10, color='b')
    ax3_twin.set_ylabel("Current [A]", fontsize=10, color='r')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
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
    plot_path = output_dir / "combined_plot.png"
    attitude_excel_path = output_dir / "姿态角.xlsx"
    rc_excel_path = output_dir / "遥控器.xlsx"
    battery_excel_path = output_dir / "电池.xlsx"
    
    if (plot_path.exists() and attitude_excel_path.exists() and 
        rc_excel_path.exists() and battery_excel_path.exists()):
        print(f"⏭️  跳过 (已处理): {log_path.name}")
        return False
    
    print(f"🔄 处理中: {log_path.name}")
    
    try:
        # 加载 ULog
        ulog = ULog(str(log_path))
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 提取数据
        attitude_df = extract_attitude(ulog)
        rc_df = extract_rc_channels(ulog)
        battery_df = extract_battery(ulog)
        
        # 保存Excel文件
        attitude_df.to_excel(attitude_excel_path, index=False)
        rc_df.to_excel(rc_excel_path, index=False)
        battery_df.to_excel(battery_excel_path, index=False)
        
        # 绘制图像
        plot_combined(attitude_df, rc_df, battery_df, plot_path)
        
        print(f"✅ 完成: {log_path.name} -> {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 错误: {log_path.name} - {str(e)}")
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
        description="批量处理PX4 ULog文件，生成姿态角、遥控器、电池数据及图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理默认ULOG文件夹，输出到Result文件夹
  python batch_parse_px4_ulog.py
  
  # 处理指定目录下的所有.ulg文件
  python batch_parse_px4_ulog.py --input-dir F:\\logs
  
  # 递归处理所有子目录中的.ulg文件
  python batch_parse_px4_ulog.py --input-dir ULOG --recursive
  
  # 指定输出目录
  python batch_parse_px4_ulog.py --input-dir ULOG --output-dir F:\\results
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd() / "ULOG",
        help="包含.ulg文件的输入目录 (默认: 当前目录/ULOG)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "Result",
        help="输出目录 (默认: 当前目录/Result)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="递归搜索子目录中的.ulg文件"
    )
    
    args = parser.parse_args()
    
    try:
        batch_process(args.input_dir, args.output_dir, args.recursive)
        input("\n按回车键退出...")
    except Exception as e:
        print(f"\n❌ 致命错误: {e}")
        input("\n按回车键退出...")
        raise


if __name__ == "__main__":
    main()
