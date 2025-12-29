% 批量处理程序：读取多个电池数据，绘制滤波前后的电压、电流和功率曲线，并汇总统计结果
clear; clc; close all;

%% 参数设置
current_coefficient = 1;  % 电流系数（可调整电流大小）
filter_window = 50;  % 滤波窗口大小（移动平均滤波）
trim_time = 15;  % 去掉前后的时间（秒）

%% 定义文件路径
% with_MTRG 数据
with_MTRG = ["ULOG_2\Result\log_0_2025-12-29-16-24-46\电池.xlsx"];

% without_MTRG 数据
without_MTRG = ["ULOG_2\Result\log_1_2025-12-29-17-23-40\电池.xlsx"];

%% 初始化汇总数据存储
summary_with_MTRG = [];
summary_without_MTRG = [];

%% 处理 with_MTRG 数据
fprintf('========================================\n');
fprintf('开始处理 with_MTRG 数据...\n');
fprintf('========================================\n\n');

for i = 1:length(with_MTRG)
    filePath = with_MTRG(i);
    fprintf('处理文件 [%d/%d]: %s\n', i, length(with_MTRG), filePath);
    
    % 处理单个文件
    [summary_row, ~] = process_single_battery_file(filePath, current_coefficient, filter_window, trim_time, sprintf('with_MTRG_%d', i));
    
    % 添加文件夹名和组别信息
    [folderPath, ~, ~] = fileparts(filePath);
    [~, folderName, ~] = fileparts(folderPath);
    summary_row = addvars(summary_row, string(folderName), 'Before', 1, 'NewVariableNames', {'文件夹名'});
    summary_row = addvars(summary_row, string('with_MTRG'), 'Before', 1, 'NewVariableNames', {'组别'});
    
    % 累积汇总数据
    summary_with_MTRG = [summary_with_MTRG; summary_row];
    
    fprintf('\n');
end

%% 处理 without_MTRG 数据
fprintf('========================================\n');
fprintf('开始处理 without_MTRG 数据...\n');
fprintf('========================================\n\n');

for i = 1:length(without_MTRG)
    filePath = without_MTRG(i);
    fprintf('处理文件 [%d/%d]: %s\n', i, length(without_MTRG), filePath);
    
    % 处理单个文件
    [summary_row, ~] = process_single_battery_file(filePath, current_coefficient, filter_window, trim_time, sprintf('without_MTRG_%d', i));
    
    % 添加文件夹名和组别信息
    [folderPath, ~, ~] = fileparts(filePath);
    [~, folderName, ~] = fileparts(folderPath);
    summary_row = addvars(summary_row, string(folderName), 'Before', 1, 'NewVariableNames', {'文件夹名'});
    summary_row = addvars(summary_row, string('without_MTRG'), 'Before', 1, 'NewVariableNames', {'组别'});
    
    % 累积汇总数据
    summary_without_MTRG = [summary_without_MTRG; summary_row];
    
    fprintf('\n');
end

%% 保存汇总统计结果
resultFolder = 'ULOG_2\Result';

% 合并两组数据
summary_all = [summary_with_MTRG; summary_without_MTRG];

% 保存合并的汇总结果
summaryFilePath_all = fullfile(resultFolder, 'endurance_analysis_summary_all.csv');
writetable(summary_all, summaryFilePath_all, 'Encoding', 'UTF-8');
fprintf('汇总统计已保存到: %s\n', summaryFilePath_all);

%% 计算统计量用于绘图
% 提取 with_MTRG 和 without_MTRG 的数据
power_with = summary_with_MTRG.('功率均值_W');
power_without = summary_without_MTRG.('功率均值_W');
time_with = summary_with_MTRG.('最大时间_s');
time_without = summary_without_MTRG.('最大时间_s');

% 计算均值和标准差
power_mean_with = mean(power_with);
power_std_with = std(power_with);
power_mean_without = mean(power_without);
power_std_without = std(power_without);

time_mean_with = mean(time_with);
time_std_with = std(time_with);
time_mean_without = mean(time_without);
time_std_without = std(time_without);

%% 绘制对比图
% 图1：功率均值对比（带误差棒）
figure('Name', 'Power Comparison', 'NumberTitle', 'off');
x = [1, 2];
y = [power_mean_without, power_mean_with];
err = [power_std_without, power_std_with];
bar(x, y, 'FaceColor', [0.3 0.6 0.9]);
hold on;
errorbar(x, y, err, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
set(gca, 'XTick', x, 'XTickLabel', {'without', 'with'});
% xlabel('Condition');
ylabel('Average power P (W)');
% title('Battery Power Comparison');
fontsize(28, "points");
box on; % 添加边框
set(gca, 'XColor', 'k', 'YColor', 'k');
hold off;

% 保存功率对比图
powerFigPath = fullfile(resultFolder, 'Power_Comparison.png');
saveas(gcf, powerFigPath);
fprintf('功率对比图已保存到: %s\n', powerFigPath);

% 图2：最大时间对比（带误差棒）
figure('Name', 'Max Time Comparison', 'NumberTitle', 'off');
x = [1, 2];
y = [time_mean_without, time_mean_with];
err = [time_std_without, time_std_with];
bar(x, y, 'FaceColor', [0.9 0.5 0.3]);
hold on;
errorbar(x, y, err, 'k.', 'LineWidth', 1.5, 'CapSize', 10);
set(gca, 'XTick', x, 'XTickLabel', {'without', 'with'});
% xlabel('Condition');
ylabel('Max time t (s)');
% title('Flight Duration Comparison');
fontsize(28, "points");
box on; % 添加边框
set(gca, 'XColor', 'k', 'YColor', 'k');
hold off;

% 保存时间对比图
timeFigPath = fullfile(resultFolder, 'Time_Comparison.png');
saveas(gcf, timeFigPath);
fprintf('时间对比图已保存到: %s\n', timeFigPath);

fprintf('\n========================================\n');
fprintf('所有数据处理完成！\n');
fprintf('========================================\n');

%% 函数定义：处理单个电池文件
function [summary_row, detail_table] = process_single_battery_file(filePath, current_coefficient, filter_window, trim_time, fig_suffix)
    % 提取文件夹路径用于保存结果
    [folderPath, ~, ~] = fileparts(filePath);
    
    % 读取数据
    data = readtable(filePath);
    
    % 提取电压和电流列
    colNames = data.Properties.VariableNames;
    
    % 尝试自动识别时间、电压和电流列
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
    
    % 应用电流系数
    current = current * current_coefficient;
    
    % 数据滤波（移动平均滤波）
    voltage_filtered = movmean(voltage, filter_window);
    current_filtered = movmean(current, filter_window);
    
    % 计算功率
    power = voltage .* current;
    power_filtered = voltage_filtered .* current_filtered;
    
    %% 绘制对比图
    figure('Name', ['Battery Data Analysis - ' fig_suffix], 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);
    
    % 子图1：电压（滤波前后对比）
    subplot(3,1,1);
    plot(time, voltage, 'b-', 'LineWidth', 1, 'DisplayName', 'Raw Voltage');
    hold on;
    plot(time, voltage_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Voltage');
    xlabel('Time (s)');
    ylabel('Voltage (V)');
    title('Battery Voltage (Raw vs Filtered)');
    legend('show', 'Location', 'best');
    fontsize(28, "points");
    box on; % 添加边框
    set(gca, 'XColor', 'k', 'YColor', 'k');
    % grid on;
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
    fontsize(28, "points");
    box on; % 添加边框
    set(gca, 'XColor', 'k', 'YColor', 'k');
    % grid on;
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
    fontsize(28, "points");
    box on; % 添加边框
    set(gca, 'XColor', 'k', 'YColor', 'k');
    % grid on;
    hold off;
    
    % 保存图片到电池.xlsx所在文件夹
    figFilePath = fullfile(folderPath, 'Battery_Data_Analysis.png');
    saveas(gcf, figFilePath);
    close(gcf);  % 关闭图形窗口以节省内存
    
    %% 计算统计信息（去掉前后trim_time秒数据）
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
    
    % 计算最大时间
    max_time = max(time) - min(time);
    
    fprintf('  原始数据点数: %d\n', length(time));
    fprintf('  有效数据点数: %d\n', sum(valid_idx));
    fprintf('  最大时间: %.2f 秒\n', max_time);
    fprintf('  电压均值: %.4f V\n', mean(voltage_stat));
    fprintf('  电流均值: %.4f A\n', mean(current_stat));
    fprintf('  功率均值: %.4f W\n', mean(power_stat));
    
    %% 保存为CSV文件
    % 1. 保存统计结果摘要（使用去除前后trim_time秒后的数据）
    summary_row = table(...
        max_time, ...
        sum(valid_idx), ...
        mean(voltage_stat), std(voltage_stat), max(voltage_stat), min(voltage_stat), ...
        mean(current_stat), std(current_stat), max(current_stat), min(current_stat), ...
        mean(power_stat), std(power_stat), max(power_stat), min(power_stat), ...
        'VariableNames', {'最大时间_s', ...
                          '有效数据点数', ...
                          '电压均值_V', '电压标准差_V', '电压最大值_V', '电压最小值_V', ...
                          '电流均值_A', '电流标准差_A', '电流最大值_A', '电流最小值_A', ...
                          '功率均值_W', '功率标准差_W', '功率最大值_W', '功率最小值_W'});
    
    summaryFilePath = fullfile(folderPath, 'endurance_analysis_summary.csv');
    writetable(summary_row, summaryFilePath, 'Encoding', 'UTF-8');
    
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
end
