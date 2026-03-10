%% ========================================================================
%  PART 3: Parallel Mandelbrot Set Computation
%  ========================================================================
%
%TODO: Implement parallel Mandelbrot set computation function
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