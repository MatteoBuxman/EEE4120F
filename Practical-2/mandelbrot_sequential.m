%% ========================================================================
%  PART 2: Serial Mandelbrot Set Computation
%  ========================================================================`
%
%TODO: Implement serial Mandelbrot set computation function
function mandelbrot_sequential(x_res, y_res, max_iterations, resolution_name, save_path)

    % --- Complex plane bounds ---
    x_range = linspace(-2, 0.5, x_res);   % real axis
    y_range = linspace(-1.2, 1.2, y_res);  % imaginary axis

    % --- Compute iteration grid ---
    iteration_grid = compute_mandelbrot(x_range, y_range, max_iterations);

    % --- Plot and save ---
    mandelbrot_plot(iteration_grid, resolution_name, save_path);
end

%Helper function for the mandelbrot_sequential function
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