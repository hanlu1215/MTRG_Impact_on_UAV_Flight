import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyulog import ULog

# ==================== 配置参数 ====================
# 设置要绘制的遥控器通道编号（1-18），可以是单个或多个通道
# 例如: [5] 表示只绘制通道5
#      [1, 2, 3, 4] 表示绘制通道1-4
RC_CHANNELS_TO_PLOT = [6]  # 修改此列表以选择不同的通道
# ==================================================

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def quaternion_to_euler(q0: Iterable[float], q1: Iterable[float], q2: Iterable[float], q3: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert quaternions to Euler angles (roll, pitch, yaw) in degrees."""
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    q3 = np.asarray(q3)

    # Formulas follow aerospace sequence ZYX.
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
    time_s = (ts - ts[0]) * 1e-6  # Convert from usec to seconds, zero-based.

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
    time_s = (ts - ts[0]) * 1e-6  # Convert from usec to seconds, zero-based.

    # Extract common RC channels (typically channels 1-4 are roll, pitch, throttle, yaw)
    rc_df = pd.DataFrame({"time_s": time_s})
    
    # Find all channel columns
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
    time_s = (ts - ts[0]) * 1e-6  # Convert from usec to seconds, zero-based.

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
        col_name = f"channel[{ch_num-1}]"  # channel[0] corresponds to channel 1
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
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sanitize_filename(name: str, index: int, suffix: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    safe = safe[:30]
    return f"{safe}_{index}{suffix}"


def export_all_topics(ulog: ULog, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    attitude_excel_path: Optional[Path] = None
    for idx, dataset in enumerate(ulog.data_list):
        if not dataset.data:
            continue
        df = pd.DataFrame(dataset.data)
        if "timestamp" in df.columns:
            df.insert(0, "timestamp", df.pop("timestamp"))
            df.insert(1, "time_s", df["timestamp"] * 1e-6)
        filename = sanitize_filename(dataset.name, idx, ".xlsx")
        file_path = output_dir / filename
        if dataset.name == "vehicle_attitude":
            attitude_excel_path = file_path
        df.to_excel(file_path, index=False)
    if attitude_excel_path:
        print(f"姿态曲线对应的Excel文件: {attitude_excel_path}")


def choose_log_file() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(
        title="选择 PX4 ULog 文件",
        filetypes=[("ULog files", "*.ulg"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse PX4 ULog, plot attitude, export topics to Excel files")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to PX4 .ulg log file; if omitted, a file dialog will open",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory; defaults to a folder named after the log file (without extension)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Path to save attitude plot (PNG)",
    )
    args = parser.parse_args()

    log_path = args.log if args.log is not None else choose_log_file()
    if log_path is None:
        raise SystemExit("未选择日志文件，已退出。")

    # ULog expects a string path or file-like; Path is not accepted directly on Windows.
    ulog = ULog(str(log_path))

    output_dir = args.outdir if args.outdir is not None else log_path.with_suffix("")
    plot_path = args.plot if args.plot is not None else output_dir / "a-combined_plot.png"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    attitude_df = extract_attitude(ulog)
    rc_df = extract_rc_channels(ulog)
    battery_df = extract_battery(ulog)

    # Save to Excel files
    attitude_excel_path = output_dir / "a-姿态角.xlsx"
    rc_excel_path = output_dir / "a-遥控器.xlsx"
    battery_excel_path = output_dir / "a-电池.xlsx"
    
    attitude_df.to_excel(attitude_excel_path, index=False)
    rc_df.to_excel(rc_excel_path, index=False)
    battery_df.to_excel(battery_excel_path, index=False)

    print(f"姿态角数据已导出: {attitude_excel_path}")
    print(f"遥控器数据已导出: {rc_excel_path}")
    print(f"电池数据已导出: {battery_excel_path}")

    # Plot combined figure
    plot_combined(attitude_df, rc_df, battery_df, plot_path)
    export_all_topics(ulog, output_dir)

    print(f"Combined plot saved to: {plot_path}")
    print(f"Per-topic Excel files saved under: {output_dir}")
    input("按回车键退出...")


if __name__ == "__main__":
    main()
