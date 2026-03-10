function plot_graphs(resolution_names, image_sizes, time_serial, time_parallel, max_workers)
    % time_parallel: [num_resolutions x (max_workers-1)]
    % col 1 = 2 workers, col 2 = 3 workers, ... col 5 = 6 workers

    num_resolutions = length(resolution_names);
    megapixels      = (image_sizes(:,1) .* image_sizes(:,2)) / 1e6;
    short_names     = {'SVGA','HD','FHD','2K','QHD','4K','5K','8K'};
    x               = 1:num_resolutions;
    worker_counts   = 2:max_workers;

    % --- One colour per worker count ---
    worker_colors = [
        0.42, 0.68, 0.98;   % 2w — blue
        0.29, 0.78, 0.60;   % 3w — green
        0.98, 0.75, 0.25;   % 4w — amber
        0.88, 0.42, 0.42;   % 5w — red
        0.75, 0.50, 0.95;   % 6w — purple
    ];

    % --- Precompute speedup and efficiency matrices ---
    % speedup(i, w-1) = time_serial(i) / time_parallel(i, w-1)
    speedup    = time_serial ./ time_parallel;
    efficiency = speedup ./ worker_counts * 100;   % broadcast division across cols

    fig = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 1400 1000]);

    % =============================================
    % Plot 1: Execution Time vs Resolution (grouped bar)
    % =============================================
    ax1 = subplot(2, 3, 1);
    bar_data = [time_serial, time_parallel];
    b = bar(x, bar_data, 'grouped');

    % Colour bars: serial = white/grey, parallel = worker colours
    b(1).FaceColor = [0.75 0.75 0.75];
    for w = 1:length(worker_counts)
        b(w+1).FaceColor = worker_colors(w, :);
    end

    legend_entries = [{'Serial'}, arrayfun(@(w) sprintf('%dw', w), worker_counts, 'UniformOutput', false)];
    lg = legend(legend_entries, 'TextColor', [0.85 0.85 0.85], ...
        'Color', [0.20 0.20 0.22], 'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8);

    style_axes(ax1, x, short_names);
    ylabel('Time (s)', 'Color', [0.85 0.85 0.85]);
    xlabel('Resolution', 'Color', [0.85 0.85 0.85]);
    title('Execution Time vs Resolution', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Plot 2: Speedup vs Resolution (one line per worker count)
    % =============================================
    ax2 = subplot(2, 3, 2);
    hold on;
    for w = 1:length(worker_counts)
        plot(x, speedup(:, w), '-o', ...
            'Color', worker_colors(w, :), ...
            'LineWidth', 2, ...
            'MarkerFaceColor', worker_colors(w, :), ...
            'MarkerSize', 6, ...
            'DisplayName', sprintf('%dw', worker_counts(w)));
    end
    % Ideal speedup lines
    for w = 1:length(worker_counts)
        yline(worker_counts(w), '--', ...
            'Color', [worker_colors(w,:), 0.3], ...
            'LineWidth', 0.8, ...
            'HandleVisibility', 'off');
    end
    hold off;
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8);
    ylim([0 max_workers + 0.5]);
    style_axes(ax2, x, short_names);
    ylabel('Speedup', 'Color', [0.85 0.85 0.85]);
    xlabel('Resolution', 'Color', [0.85 0.85 0.85]);
    title('Speedup vs Resolution', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Plot 3: Efficiency vs Resolution
    % =============================================
    ax3 = subplot(2, 3, 3);
    hold on;
    for w = 1:length(worker_counts)
        plot(x, efficiency(:, w), '-s', ...
            'Color', worker_colors(w, :), ...
            'LineWidth', 2, ...
            'MarkerFaceColor', worker_colors(w, :), ...
            'MarkerSize', 6, ...
            'DisplayName', sprintf('%dw', worker_counts(w)));
    end
    yline(100, '--', 'Ideal', 'Color', [0.9 0.9 0.9], 'LineWidth', 1.2, ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 8, 'HandleVisibility', 'off');
    hold off;
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8);
    ylim([0 120]);
    style_axes(ax3, x, short_names);
    ylabel('Efficiency (%)', 'Color', [0.85 0.85 0.85]);
    xlabel('Resolution', 'Color', [0.85 0.85 0.85]);
    title('Parallel Efficiency vs Resolution', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Plot 4: Time vs Megapixels log scale
    % =============================================
    ax4 = subplot(2, 3, 4);
    hold on;
    semilogy(megapixels, time_serial, '-o', ...
        'Color', [0.75 0.75 0.75], 'LineWidth', 2, ...
        'MarkerFaceColor', [0.75 0.75 0.75], 'MarkerSize', 6, 'DisplayName', 'Serial');
    for w = 1:length(worker_counts)
        semilogy(megapixels, time_parallel(:, w), '-o', ...
            'Color', worker_colors(w, :), 'LineWidth', 2, ...
            'MarkerFaceColor', worker_colors(w, :), 'MarkerSize', 6, ...
            'DisplayName', sprintf('%dw', worker_counts(w)));
    end
    hold off;
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8);
    style_axes(ax4, [], []);
    ax4.YGrid = 'on';
    xlabel('Megapixels', 'Color', [0.85 0.85 0.85]);
    ylabel('Time (s) — log scale', 'Color', [0.85 0.85 0.85]);
    title('Scaling: Time vs Megapixels', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Plot 5: Speedup vs Worker Count (one line per resolution)
    % =============================================
    ax5 = subplot(2, 3, 5);
    res_colors = cool(num_resolutions);
    hold on;
    for i = 1:num_resolutions
        plot(worker_counts, speedup(i, :), '-o', ...
            'Color', res_colors(i, :), 'LineWidth', 1.8, ...
            'MarkerFaceColor', res_colors(i, :), 'MarkerSize', 5, ...
            'DisplayName', short_names{i});
    end
    % Ideal linear speedup
    plot(worker_counts, worker_counts, 'w--', 'LineWidth', 1.2, ...
        'DisplayName', 'Ideal', 'HandleVisibility', 'on');
    hold off;
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8, 'NumColumns', 2);
    style_axes(ax5, worker_counts, arrayfun(@num2str, worker_counts, 'UniformOutput', false));
    xlabel('Number of Workers', 'Color', [0.85 0.85 0.85]);
    ylabel('Speedup', 'Color', [0.85 0.85 0.85]);
    title('Speedup vs Worker Count', 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Plot 6: Amdahl's Law — estimated serial fraction
    % =============================================
    ax6 = subplot(2, 3, 6);
    % Estimate serial fraction f from measured speedup at max workers
    % From Amdahl: S = 1/((1-f) + f/P) => f = (1/S - 1) / (1/P - 1)
    P = max_workers;
    S = speedup(:, end);   % speedup at max workers
    f = (1./S - 1) ./ (1/P - 1);
    f = max(0, min(1, f));  % clamp to [0,1]

    barh(x, f * 100, 'FaceColor', [0.42 0.68 0.98], 'EdgeColor', 'none');
    xline(0, 'Color', [0.5 0.5 0.5]);
    style_axes(ax6, x, short_names);
    ax6.YTickLabel = short_names;
    ax6.XColor     = [0.85 0.85 0.85];
    xlabel('Estimated Serial Fraction (%)', 'Color', [0.85 0.85 0.85]);
    title("Amdahl's Law — Serial Fraction", 'Color', 'white', 'FontSize', 10, 'FontWeight', 'bold');

    % =============================================
    % Title + save
    % =============================================
    sgtitle('Mandelbrot Set — Serial vs Parallel Performance Analysis', ...
        'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold');

    if ~exist('output', 'dir'), mkdir('output'); end
    print(fig, 'output/performance_analysis.png', '-dpng', '-r150');
    fprintf('Saved: output/performance_analysis.png\n');
end

% =========================================================
% Helper: apply consistent dark theme styling to an axes
% =========================================================
function style_axes(ax, x_ticks, x_labels)
    ax.Color      = [0.16 0.16 0.18];
    ax.XColor     = [0.85 0.85 0.85];
    ax.YColor     = [0.85 0.85 0.85];
    ax.GridColor  = [0.28 0.28 0.30];
    ax.YGrid      = 'on';
    ax.XGrid      = 'off';
    ax.FontSize   = 9;
    ax.Box        = 'off';
    if ~isempty(x_ticks)
        ax.XTick       = x_ticks;
        ax.XTickLabel  = x_labels;
    end
end