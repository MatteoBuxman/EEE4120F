function profile_load_balance(num_workers)
% PROFILE_LOAD_BALANCE  Run an instrumented parfor on 4K UHD and plot
%   the per-worker workload distribution.
%
%   profile_load_balance(num_workers)
%
%   Produces two plots saved to output/:
%     1. Per-row computation time coloured by assigned worker
%     2. Total computation time per worker (bar chart)

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
