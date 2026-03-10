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
function run_analysis()

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

end