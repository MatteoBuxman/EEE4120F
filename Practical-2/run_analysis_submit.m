% =========================================================================
% Practical 2: Mandelbrot-Set Serial vs Parallel Analysis
% =========================================================================
%
% GROUP NUMBER:
%
% MEMBERS:
%   - Matteo Buxman, BXMMAT001
%   - Emmanuel Basua, BSXEMM001
%% ========================================================================
%  PART 4: Analyse the runtime of both the serial and the parallel
%  implementation
%  ========================================================================
%

%AI was used to help with implementing the graph plotting routines, as well as to collect all the subroutines
%into this single file for the purposes of submission.

function run_analysis_submit()

    image_sizes = [
        800,  600;   % SVGA
        1280,  720;  % HD
        1920, 1080;  % Full HD
        2048, 1080;  % 2K Cinema
        2560, 1440;  % 2K QHD
        3840, 2160;  % 4K UHD
        5120, 2880;  % 5K
        7680, 4320   % 8K UHD
    ];

    resolution_names = {'SVGA', 'HD', 'Full HD', '2K', 'QHD', '4K UHD', '5K', '8K UHD'};
    max_iterations   = 1000;
    max_workers      = 6;       % physical cores on this machine
    num_resolutions  = size(image_sizes, 1);

    % --- Preallocate timing arrays ---
    % time_parallel: rows = resolutions, cols = worker counts (2,3,4,5,6)
    time_serial   = zeros(num_resolutions, 1);
    time_parallel = zeros(num_resolutions, max_workers - 1);

    % =============================================
    % SERIAL RUNS
    % =============================================
    fprintf('\n=== SERIAL ===\n');
    for i = 1:num_resolutions
        width  = image_sizes(i, 1);
        height = image_sizes(i, 2);
        name   = resolution_names{i};

        t_start        = tic;
        mandelbrot_sequential(width, height, max_iterations, name, ...
            sprintf('output/serial/%s.png', name));
        time_serial(i) = toc(t_start);

        fprintf('[Serial] %s (%dx%d): %.2f sec\n', name, width, height, time_serial(i));
    end

    % =============================================
    % PARALLEL RUNS — sweep 2 to max_workers
    % =============================================
    for num_workers = 2:max_workers
        fprintf('\n=== PARALLEL (%d workers) ===\n', num_workers);

        % --- Spin up pool BEFORE timing starts ---
        pool = gcp('nocreate');
        if isempty(pool)
            parpool(num_workers);
        elseif pool.NumWorkers ~= num_workers
            delete(pool);
            parpool(num_workers);
        end

        for i = 1:num_resolutions
            width  = image_sizes(i, 1);
            height = image_sizes(i, 2);
            name   = resolution_names{i};

            save_path = sprintf('output/parallel_%dw/%s.png', num_workers, name);

            t_start                            = tic;
            mandelbrot_parallel(width, height, max_iterations, name, ...
                save_path, num_workers);
            time_parallel(i, num_workers - 1)  = toc(t_start);

            speedup = time_serial(i) / time_parallel(i, num_workers - 1);
            fprintf('[Parallel %dw] %s (%dx%d): %.2f sec | Speedup: %.2fx\n', ...
                num_workers, name, width, height, ...
                time_parallel(i, num_workers - 1), speedup);
        end
    end

    % =============================================
    % SUMMARY TABLE
    % =============================================
    fprintf('\n=== SUMMARY ===\n');
    fprintf('%-12s %10s', 'Resolution', 'Serial(s)');
    for w = 2:max_workers
        fprintf(' %8s', sprintf('Par_%dw(s)', w));
    end
    fprintf('  BestSpeedup  BestEfficiency\n');

    for i = 1:num_resolutions
        fprintf('%-12s %10.2f', resolution_names{i}, time_serial(i));
        for w = 2:max_workers
            fprintf(' %12.2f', time_parallel(i, w - 1));
        end
        [best_time, best_idx] = min(time_parallel(i, :));
        best_workers   = best_idx + 1;
        best_speedup   = time_serial(i) / best_time;
        best_efficiency = (best_speedup / best_workers) * 100;
        fprintf('  %.2fx        %.1f%%\n', best_speedup, best_efficiency);
    end

    % =============================================
    % GRAPHS
    % =============================================
    plot_graphs(resolution_names, image_sizes, time_serial, time_parallel, max_workers);

    % =============================================
    % LOAD BALANCE PROFILING (uses existing pool)
    % =============================================
    profile_load_balance(max_workers);

end

%% ========================================================================
%  PART 1: Mandelbrot Set Image Plotting and Saving
%  ========================================================================
function mandelbrot_plot(image_data, resolution_name, save_path)

    [dir_path, ~, ~] = fileparts(save_path);
    if ~isempty(dir_path) && ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end

    % Map iteration counts directly to pixels via hot colormap
    normalized = mat2gray(image_data);
    cmap        = hot(256);
    indices     = gray2ind(normalized, 256);
    rgb         = ind2rgb(indices, cmap);

    imwrite(rgb, save_path);

    [height, width] = size(image_data);
    fprintf('Saved %s: %dx%d px (%.2f MP) -> %s\n', ...
        resolution_name, width, height, (width*height)/1e6, save_path);
end

%% ========================================================================
%  PART 2: Serial Mandelbrot Set Computation
%  ========================================================================
function mandelbrot_sequential(x_res, y_res, max_iterations, resolution_name, save_path)

    % --- Complex plane bounds ---
    x_range = linspace(-2, 0.5, x_res);   % real axis
    y_range = linspace(-1.2, 1.2, y_res);  % imaginary axis

    % --- Compute iteration grid ---
    iteration_grid = compute_mandelbrot(x_range, y_range, max_iterations);

    % --- Plot and save ---
    mandelbrot_plot(iteration_grid, resolution_name, save_path);
end

% Helper function for the mandelbrot_sequential function
function iteration_grid = compute_mandelbrot(x_range, y_range, max_iterations)
    % Computes the escape-time iteration count for each point in the grid

    iteration_grid = zeros(length(y_range), length(x_range));

    for row = 1:length(y_range)
        for col = 1:length(x_range)

            x0 = x_range(col);   % real part of c
            y0 = y_range(row);   % imaginary part of c

            x = 0;
            y = 0;
            iteration = 0;

            while (iteration < max_iterations && x^2 + y^2 <= 4)
                x_next = x^2 - y^2 + x0;
                y_next = 2*x*y + y0;

                x = x_next;
                y = y_next;
                iteration = iteration + 1;
            end

            iteration_grid(row, col) = iteration;
        end
    end
end

%% ========================================================================
%  PART 3: Parallel Mandelbrot Set Computation
%  ========================================================================
function mandelbrot_parallel(x_res, y_res, max_iterations, resolution_name, save_path, num_workers)

    % --- Complex plane bounds ---
    x_range = linspace(-2, 0.5, x_res);
    y_range = linspace(-1.2, 1.2, y_res);

    % --- Preallocate ---
    iteration_grid = zeros(length(y_range), length(x_range));

    % --- Parallel outer loop over rows ---
    parfor row = 1:length(y_range)
        y0 = y_range(row);                          % broadcast scalar
        row_result = zeros(1, length(x_range));     % sliced output

        for col = 1:length(x_range)
            x0 = x_range(col);

            x = 0;
            y = 0;
            iteration = 0;

            while (iteration < max_iterations && x^2 + y^2 <= 4)
                x_next = x^2 - y^2 + x0;
                y_next = 2*x*y + y0;
                x = x_next;
                y = y_next;
                iteration = iteration + 1;
            end

            row_result(col) = iteration;
        end

        iteration_grid(row, :) = row_result;        % sliced assignment
    end

    % --- Plot and save ---
    mandelbrot_plot(iteration_grid, resolution_name, save_path);
end

%% ========================================================================
%  Performance Graphs
%  ========================================================================
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

%% ========================================================================
%  Load Balance Profiling
%  ========================================================================
function profile_load_balance(num_workers)

    if nargin < 1, num_workers = 6; end

    % --- Resolution: 4K UHD (large enough to show clear patterns) ---
    x_res = 3840;
    y_res = 2160;
    max_iterations = 1000;

    x_range = linspace(-2, 0.5, x_res);
    y_range = linspace(-1.2, 1.2, y_res);

    row_times  = zeros(y_res, 1);
    worker_ids = zeros(y_res, 1);

    % --- Ensure pool is ready ---
    pool = gcp('nocreate');
    if isempty(pool)
        parpool(num_workers);
    elseif pool.NumWorkers ~= num_workers
        delete(pool);
        parpool(num_workers);
    end

    fprintf('Profiling 4K UHD (%dx%d) with %d workers...\n', x_res, y_res, num_workers);

    % --- Instrumented parfor ---
    parfor row = 1:y_res
        t_row = tic;
        y0 = y_range(row);
        row_result = zeros(1, x_res);

        for col = 1:x_res
            x0 = x_range(col);
            x = 0; y = 0; iteration = 0;
            while (iteration < max_iterations && x^2 + y^2 <= 4)
                x_next = x^2 - y^2 + x0;
                y_next = 2*x*y + y0;
                x = x_next; y = y_next;
                iteration = iteration + 1;
            end
            row_result(col) = iteration;
        end

        row_times(row) = toc(t_row);

        w = getCurrentWorker();
        worker_ids(row) = w.ProcessId;
    end

    % --- Map process IDs to sequential 1:N labels ---
    unique_pids = unique(worker_ids);
    worker_labels = zeros(y_res, 1);
    for k = 1:length(unique_pids)
        worker_labels(worker_ids == unique_pids(k)) = k;
    end

    if ~exist('output', 'dir'), mkdir('output'); end

    % --- Colours per worker ---
    cmap = lines(num_workers);

    % =============================================
    % Plot 1: Per-row computation time by worker
    % =============================================
    fig1 = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 800 400]);
    ax1 = axes(fig1);
    hold on;
    for k = 1:num_workers
        mask = (worker_labels == k);
        rows_k = find(mask);
        scatter(rows_k, row_times(mask), 4, cmap(k,:), 'filled', ...
            'DisplayName', sprintf('Worker %d', k));
    end
    hold off;
    ax1.Color     = [0.16 0.16 0.18];
    ax1.XColor    = [0.85 0.85 0.85];
    ax1.YColor    = [0.85 0.85 0.85];
    ax1.GridColor = [0.28 0.28 0.30];
    ax1.YGrid     = 'on';
    ax1.Box       = 'off';
    ax1.FontSize  = 9;
    xlabel('Row index', 'Color', [0.85 0.85 0.85]);
    ylabel('Computation time (s)', 'Color', [0.85 0.85 0.85]);
    title('Per-Row Computation Time by Worker (4K UHD)', ...
        'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    legend('TextColor', [0.85 0.85 0.85], 'Color', [0.20 0.20 0.22], ...
        'EdgeColor', [0.35 0.35 0.35], 'FontSize', 8, 'NumColumns', 3);
    print(fig1, 'output/load_balance_per_row.png', '-dpng', '-r150');
    fprintf('Saved: output/load_balance_per_row.png\n');

    % =============================================
    % Plot 2: Total work per worker (bar chart)
    % =============================================
    total_time = zeros(num_workers, 1);
    num_rows   = zeros(num_workers, 1);
    for k = 1:num_workers
        total_time(k) = sum(row_times(worker_labels == k));
        num_rows(k)   = sum(worker_labels == k);
    end

    fig2 = figure('Color', [0.10 0.10 0.12], 'Position', [50 50 600 400]);
    ax2 = axes(fig2);
    b = bar(1:num_workers, total_time, 'FaceColor', 'flat', 'EdgeColor', 'none');
    for k = 1:num_workers
        b.CData(k,:) = cmap(k,:);
    end
    % Add row-count labels on each bar
    for k = 1:num_workers
        text(k, total_time(k) + max(total_time)*0.02, ...
            sprintf('%d rows', num_rows(k)), ...
            'HorizontalAlignment', 'center', 'Color', [0.85 0.85 0.85], ...
            'FontSize', 8);
    end
    ax2.Color     = [0.16 0.16 0.18];
    ax2.XColor    = [0.85 0.85 0.85];
    ax2.YColor    = [0.85 0.85 0.85];
    ax2.GridColor = [0.28 0.28 0.30];
    ax2.YGrid     = 'on';
    ax2.Box       = 'off';
    ax2.FontSize  = 9;
    ax2.XTick     = 1:num_workers;
    ax2.XTickLabel = arrayfun(@(k) sprintf('W%d', k), 1:num_workers, 'UniformOutput', false);
    xlabel('Worker', 'Color', [0.85 0.85 0.85]);
    ylabel('Total computation time (s)', 'Color', [0.85 0.85 0.85]);
    title('Total Workload per Worker (4K UHD)', ...
        'Color', 'white', 'FontSize', 12, 'FontWeight', 'bold');
    print(fig2, 'output/load_balance_total.png', '-dpng', '-r150');
    fprintf('Saved: output/load_balance_total.png\n');

    % --- Print summary ---
    fprintf('\n=== LOAD BALANCE SUMMARY ===\n');
    for k = 1:num_workers
        fprintf('Worker %d: %4d rows, %.2f s total, %.4f s/row avg\n', ...
            k, num_rows(k), total_time(k), total_time(k)/num_rows(k));
    end
    imbalance = (max(total_time) - min(total_time)) / mean(total_time) * 100;
    fprintf('Imbalance: %.1f%% (max-min as %% of mean)\n', imbalance);
end
