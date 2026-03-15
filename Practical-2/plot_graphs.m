function plot_graphs(resolution_names, image_sizes, time_serial, time_parallel, max_workers)
    % time_parallel: [num_resolutions x (max_workers-1)]
    % col 1 = 2 workers, col 2 = 3 workers, ... col 5 = 6 workers

    num_resolutions = length(resolution_names);
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
    speedup    = time_serial ./ time_parallel;
    efficiency = speedup ./ worker_counts * 100;

    % --- Estimate serial fraction per resolution (Amdahl's Law) ---
    % Rearranged: f = (1/S - 1/P) / (1 - 1/P)
    serial_frac = zeros(num_resolutions, 1);
    for i = 1:num_resolutions
        f_estimates = (1./speedup(i,:) - 1./worker_counts) ./ (1 - 1./worker_counts);
        serial_frac(i) = mean(max(0, min(1, f_estimates)));
    end

    res_colors = cool(num_resolutions);

    if ~exist('output', 'dir'), mkdir('output'); end

    % =============================================
    % Plot 1: Speedup vs Worker Count with Amdahl's Law overlay
    % =============================================
    fig1 = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 700 500]);
    ax1 = axes(fig1);
    hold on;
    for i = 1:num_resolutions
        plot(worker_counts, speedup(i, :), '-o', ...
            'Color', res_colors(i, :), 'LineWidth', 1.8, ...
            'MarkerFaceColor', res_colors(i, :), 'MarkerSize', 5, ...
            'DisplayName', short_names{i});
    end
    % Amdahl's Law theoretical curves — dashed, per resolution
    P_fine = linspace(2, max_workers, 50);
    for i = 1:num_resolutions
        f = serial_frac(i);
        S_theory = 1 ./ (f + (1 - f) ./ P_fine);
        plot(P_fine, S_theory, '--', 'Color', [res_colors(i,:), 0.4], ...
            'LineWidth', 1.2, 'HandleVisibility', 'off');
    end
    plot(worker_counts, worker_counts, 'w--', 'LineWidth', 1.2, ...
        'DisplayName', 'Ideal');
    hold off;
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8, 'NumColumns', 3, ...
        'Location', 'northwest');
    style_axes(ax1, worker_counts, arrayfun(@num2str, worker_counts, 'UniformOutput', false));
    xlabel('Number of Workers', 'Color', [0.85 0.85 0.85]);
    ylabel('Speedup', 'Color', [0.85 0.85 0.85]);
    title('Speedup vs Workers (with Amdahl''s Law)', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    print(fig1, 'output/speedup_vs_workers.png', '-dpng', '-r150');
    fprintf('Saved: output/speedup_vs_workers.png\n');

    % =============================================
    % Plot 2: Efficiency vs Resolution
    % =============================================
    fig2 = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 700 500]);
    ax2 = axes(fig2);
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
    style_axes(ax2, x, short_names);
    ylabel('Efficiency (%)', 'Color', [0.85 0.85 0.85]);
    xlabel('Resolution', 'Color', [0.85 0.85 0.85]);
    title('Parallel Efficiency vs Resolution', 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    print(fig2, 'output/efficiency_vs_resolution.png', '-dpng', '-r150');
    fprintf('Saved: output/efficiency_vs_resolution.png\n');

    % =============================================
    % Plot 3: Amdahl's Law — estimated serial fraction
    % =============================================
    fig3 = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 700 500]);
    ax3 = axes(fig3);
    barh(x, serial_frac * 100, 'FaceColor', [0.42 0.68 0.98], 'EdgeColor', 'none');
    xline(0, 'Color', [0.5 0.5 0.5]);
    style_axes(ax3, [], {});
    ax3.YTick      = x;
    ax3.YTickLabel = short_names;
    xlabel('Estimated Serial Fraction (%)', 'Color', [0.85 0.85 0.85]);
    title("Amdahl's Law — Serial Fraction", 'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    print(fig3, 'output/amdahls_law_serial_fraction.png', '-dpng', '-r150');
    fprintf('Saved: output/amdahls_law_serial_fraction.png\n');
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
