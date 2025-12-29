% 主程序：读取电池数据，绘制滤波前后的电压、电流和功率曲线
clear; clc; close all;

%% 参数设置
current_coefficient =1;  % 电流系数（可调整电流大小）
filter_window = 50;  % 滤波窗口大小（移动平均滤波）

%% 读取Excel数据
filePath = 'ULOG_2\Result\log_0_2025-12-29-16-24-46\电池.xlsx';

% 提取文件夹路径用于保存结果
[folderPath, ~, ~] = fileparts(filePath);

% 读取数据
data = readtable(filePath);

% 提取电压和电流列（根据实际列名调整）
% 假设第一列是时间，第二列是电压，第三列是电流
% 如果列名不同，请调整下面的代码
colNames = data.Properties.VariableNames;
disp('Excel文件中的列名：');
disp(colNames);

% 尝试自动识别时间、电压和电流列
% 这里假设标准的列名，您可能需要根据实际情况调整
if width(data) >= 3
    time = data{:,1};
    voltage = data{:,2};
    current = data{:,3};
elseif width(data) >= 2
    voltage = data{:,1};
    current = data{:,2};
    time = (1:length(voltage))';
else
    error('数据列数不足');
end

%   应用电流系数
current = current * current_coefficient;
fprintf('电流系数: %.4f\n', current_coefficient);

%% 数据滤波（移动平均滤波）
voltage_filtered = movmean(voltage, filter_window);
current_filtered = movmean(current, filter_window);

% 计算功率
power = voltage .* current;
power_filtered = voltage_filtered .* current_filtered;

fprintf('滤波窗口大小: %d\n', filter_window);
fprintf('滤波窗口大小: %d\n', filter_window);

%% 绘制对比图
figure('Name', 'Battery Data Analysis', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% 子图1：电压（滤波前后对比）
subplot(3,1,1);
plot(time, voltage, 'b-', 'LineWidth', 1, 'DisplayName', 'Raw Voltage');
hold on;
plot(time, voltage_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Voltage');
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Battery Voltage (Raw vs Filtered)');
legend('show', 'Location', 'best');
grid on;
hold off;

% 子图2：电流（滤波前后对比）
subplot(3,1,2);
plot(time, current, 'b-', 'LineWidth', 1, 'DisplayName', 'Raw Current');
hold on;
plot(time, current_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Current');
xlabel('Time (s)');
ylabel('Current (A)');
title('Battery Current (Raw vs Filtered)');
legend('show', 'Location', 'best');
grid on;
hold off;

% 子图3：功率（滤波前后对比）
subplot(3,1,3);
plot(time, power, 'b-', 'LineWidth', 1, 'DisplayName', 'Raw Power');
hold on;
plot(time, power_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Power');
xlabel('Time (s)');
ylabel('Power (W)');
title('Battery Power (Raw vs Filtered)');
legend('show', 'Location', 'best');
grid on;
hold off;

% 保存图片到电池.xlsx所在文件夹
figFilePath = fullfile(folderPath, 'Battery_Data_Analysis.png');
saveas(gcf, figFilePath);
fprintf('图片已保存到: %s\n', figFilePath);

%% 计算统计信息（去掉前后15秒数据）
% 定义要去掉的时间（秒）
trim_time = 15;

% 找到时间范围
time_min = min(time);
time_max = max(time);
time_start = time_min + trim_time;
time_end = time_max - trim_time;

% 创建索引掩码，选择中间部分数据
valid_idx = (time >= time_start) & (time <= time_end);

% 提取有效数据用于统计
voltage_stat = voltage_filtered(valid_idx);
current_stat = current_filtered(valid_idx);
power_stat = power_filtered(valid_idx);

fprintf('\n===================================\n');
fprintf('数据统计信息（滤波后，去掉前后%d秒）：\n', trim_time);
fprintf('===================================\n');
fprintf('原始数据点数: %d\n', length(time));
fprintf('有效数据点数: %d\n', sum(valid_idx));
fprintf('统计时间范围: %.2f - %.2f 秒\n', time_start, time_end);
fprintf('\n电压统计:\n');
fprintf('  均值: %.4f V\n', mean(voltage_stat));
fprintf('  标准差: %.4f V\n', std(voltage_stat));
fprintf('  最大值: %.4f V\n', max(voltage_stat));
fprintf('  最小值: %.4f V\n', min(voltage_stat));
fprintf('\n电流统计:\n');
fprintf('  均值: %.4f A\n', mean(current_stat));
fprintf('  标准差: %.4f A\n', std(current_stat));
fprintf('  最大值: %.4f A\n', max(current_stat));
fprintf('  最小值: %.4f A\n', min(current_stat));
fprintf('\n功率统计:\n');
fprintf('  均值: %.4f W\n', mean(power_stat));
fprintf('  标准差: %.4f W\n', std(power_stat));
fprintf('  最大值: %.4f W\n', max(power_stat));
fprintf('  最小值: %.4f W\n', min(power_stat));
fprintf('===================================\n');

fprintf('===================================\n');

%% 保存为CSV文件
% 1. 保存统计结果摘要（使用去除前后15秒后的数据）
summary_table = table(...
    sum(valid_idx), ...
    mean(voltage_stat), std(voltage_stat), max(voltage_stat), min(voltage_stat), ...
    mean(current_stat), std(current_stat), max(current_stat), min(current_stat), ...
    mean(power_stat), std(power_stat), max(power_stat), min(power_stat), ...
    'VariableNames', {'有效数据点数', ...
                      '电压均值_V', '电压标准差_V', '电压最大值_V', '电压最小值_V', ...
                      '电流均值_A', '电流标准差_A', '电流最大值_A', '电流最小值_A', ...
                      '功率均值_W', '功率标准差_W', '功率最大值_W', '功率最小值_W'});
summaryFilePath = fullfile(folderPath, 'endurance_analysis_summary.csv');
writetable(summary_table, summaryFilePath, 'Encoding', 'UTF-8');
fprintf('统计摘要已保存到: %s（已去除前后15秒数据）\n', summaryFilePath);

% 2. 保存完整数据（原始数据和滤波后数据）
detail_table = table(time, ...
    voltage, voltage_filtered, ...
    current, current_filtered, ...
    power, power_filtered, ...
    'VariableNames', {'时间', ...
                      '电压_原始_V', '电压_滤波_V', ...
                      '电流_原始_A', '电流_滤波_A', ...
                      '功率_原始_W', '功率_滤波_W'});
detailFilePath = fullfile(folderPath, 'endurance_analysis_detail.csv');
writetable(detail_table, detailFilePath, 'Encoding', 'UTF-8');
fprintf('详细数据已保存到: %s\n', detailFilePath);
