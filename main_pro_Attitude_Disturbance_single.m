%% 姿态扰动数据处理与可视化
% 功能：读取并绘制姿态角、遥控器和电池数据
% 参考：batch_parse_px4_ulog_Attitude_Disturbance.py 中的 combined_plot.png
hll
%% ==================== 配置参数 ====================
% 设置数据目录（便于修改）
data_dir = 'ULOG_1\Result\log_27_2025-12-28-22-45-50';

% 遥控器通道选择（对应Python中的RC_CHANNELS_TO_PLOT = [6]）
rc_channels_to_plot = [6];  % 可以设置为多个通道，例如 [1, 2, 3, 4]
% ==================================================

%% 读取Excel数据
fprintf('正在读取数据...\n');

% 读取姿态角数据
attitude_file = fullfile(data_dir, '姿态角.xlsx');
attitude_data = readtable(attitude_file);

% 读取遥控器数据
rc_file = fullfile(data_dir, '遥控器.xlsx');
rc_data = readtable(rc_file);

% 读取电池数据
battery_file = fullfile(data_dir, '电池.xlsx');
battery_data = readtable(battery_file);

fprintf('数据读取完成！\n');

%% 检测遥控器通道6的跳变点
col_name = 'channel_5_';  % 通道6对应的列名（从0开始索引）
col_idx = find(contains(rc_data.Properties.VariableNames, col_name));

if ~isempty(col_idx)
    channel6_data = rc_data{:, col_idx(1)};
    time_data = rc_data.time_s;
    
    % 计算差分以检测跳变
    diff_data = [0; diff(channel6_data)];
    
    % 设置阈值来识别显著跳变（可根据实际情况调整）
    threshold = 100;  % 阈值，单位为微秒
    
    % 找到跳变点的索引
    jump_indices = find(abs(diff_data) > threshold);
    
    % 获取跳变点的时间
    jump_times = time_data(jump_indices);
    
    fprintf('检测到 %d 个跳变点\n', length(jump_times));
else
    jump_times = [];
    fprintf('未找到通道6数据\n');
end

%% 绘制综合图表（3个子图）
figure('Position', [100, 100, 1500,800]);

%% 子图1: 遥控器通道
subplot(3, 1, 1);
hold on; 

% 遍历指定的遥控器通道
for i = 1:length(rc_channels_to_plot)
    ch_num = rc_channels_to_plot(i);
    % MATLAB表格中的列名格式为 'channel_x__'（其中x从0开始）
    col_name = sprintf('channel_%d_', ch_num - 1);
    
    % 查找匹配的列
    col_idx = find(contains(rc_data.Properties.VariableNames, col_name));
    
    if ~isempty(col_idx)
        plot(rc_data.time_s, rc_data{:, col_idx(1)}, 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%d', ch_num));
    else
        warning('未找到通道 %d 的数据', ch_num);
    end
end

% title('RC Channels', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Ch6 t (us)', 'FontSize', 10);
% legend('Location', 'northeast');

% 绘制跳变点的竖线
for i = 1:length(jump_times)
    xline(jump_times(i), 'k--', 'LineWidth', 1.5);
end


fontsize(28, "points");
box on;
set(gca, 'XColor', 'k', 'YColor', 'k');

hold off;

%% 子图2: 姿态角
subplot(3, 1, 2);
hold on; 

plot(attitude_data.time_s, attitude_data.roll_deg, 'LineWidth', 1.5, 'DisplayName', 'Roll');
plot(attitude_data.time_s, attitude_data.pitch_deg, 'LineWidth', 1.5, 'DisplayName', 'Pitch');
% plot(attitude_data.time_s, attitude_data.yaw_deg, 'LineWidth', 1.5, 'DisplayName', 'Yaw');

% title('Attitude Angles', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Angle (deg)', 'FontSize', 10);
ylim([-5, 5]);
legend('Location', 'northeast', 'Orientation', 'horizontal');

% 绘制跳变点的竖线
for i = 1:length(jump_times)
    xline(jump_times(i), 'k--', 'LineWidth', 1.5,'HandleVisibility','off');
end

fontsize(28, "points");
box on;
set(gca, 'XColor', 'k', 'YColor', 'k');

hold off;

%% 子图3: 电池电压和电流（双Y轴）
subplot(3, 1, 3);

% 左Y轴: 电压
yyaxis left
plot(battery_data.time_s, battery_data.voltage_v, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Voltage');
ylabel('Voltage V (V)', 'FontSize', 10, 'Color', 'b');
ax = gca;
ax.YColor = 'b';

% 右Y轴: 电流
yyaxis right
plot(battery_data.time_s, battery_data.current_a, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Current');
ylabel('Current I (A)', 'FontSize', 10, 'Color', 'r');
ax.YColor = 'r';

% title('Battery Status', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Time t (s)', 'FontSize', 10);

% 绘制跳变点的竖线
for i = 1:length(jump_times)
    xline(jump_times(i), 'k--', 'LineWidth', 1.5);
end


fontsize(28, "points");
box on;
set(gca, 'XColor', 'k', 'YColor', 'k');

% 添加图例（需要手动组合双Y轴的图例）
yyaxis left
hold on;
h1 = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
yyaxis right
h2 = plot(NaN, NaN, 'r-', 'LineWidth', 1.5);
legend([h1, h2], {'Voltage', 'Current'}, 'Location', 'east', 'Orientation', 'vertical');
hold off;

%% 保存图像
output_file = fullfile(data_dir, 'combined_plot_matlab.png');
print(gcf, output_file, '-dpng', '-r200');

fprintf('图像已保存至: %s\n', output_file);
fprintf('绘图完成！\n');
