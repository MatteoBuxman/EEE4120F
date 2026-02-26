% =========================================================================
% Practical 1: 2D Convolution Analysis
% =========================================================================
%
% GROUP NUMBER: 
%
% MEMBERS:
%   - Emmanuel Basua, BSXEMM001
%   - Matteo Buxman, BXMMAT001


%% ========================================================================
%  PART 3: Testing and Analysis
%  ========================================================================
% Moved Part 3 up due to MATLAB function ordering requirements

function run_analysis()
    % Configuration
    img_folder = 'sample_images/';
    img_list = {'image_128x128.png', 'image_256x256.png', 'image_512x512.png', 'image_1024x1024.png', 'image_2048x2048.png'};
    num_runs = 5;  % Number of runs for timing accuracy

    % Initialize results storage structure
    results = struct();
    results.image_names = {};
    results.image_sizes = [];
    results.manual_times = [];
    results.builtin_times = [];
    results.speedups = [];
    results.matches = [];

    % Print header
    fprintf('\n========== 2D Convolution Performance Analysis ==========\n');
    fprintf('Number of runs per image: %d\n\n', num_runs);
    fprintf('%-20s | %-12s | %-12s | %-8s | %-8s\n', 'Image Name', 'Manual (s)', 'Builtin (s)', 'Speedup', 'Match');
    fprintf('%s\n', repmat('-', 1, 70));

    for k = 1:length(img_list)
        img_path = fullfile(img_folder, img_list{k});
        if ~exist(img_path, 'file')
            fprintf('Warning: %s not found, skipping...\n', img_list{k});
            continue;
        end

        % Get image size for results storage
        img_info = imfinfo(img_path);
        img_size = img_info.Width;

        % Multiple runs for timing accuracy
        manual_times = zeros(1, num_runs);
        builtin_times = zeros(1, num_runs);

        for run = 1:num_runs
            % Measure Manual execution
            tic;
            out_manual = my_conv2(img_path);
            manual_times(run) = toc;

            % Measure Built-in execution
            tic;
            out_builtin = inbuilt_conv2(img_path);
            builtin_times(run) = toc;
        end

        % Calculate average times (excluding first run for warm-up)
        if num_runs > 1
            t_manual = mean(manual_times(2:end));
            t_builtin = mean(builtin_times(2:end));
        else
            t_manual = manual_times(1);
            t_builtin = builtin_times(1);
        end

        % Calculate speedup and verify correctness
        speedup = t_manual / t_builtin;
        difference = max(abs(out_manual(:) - out_builtin(:)));
        is_correct = difference < 1e-10;

        % Store results
        results.image_names{end+1} = img_list{k};
        results.image_sizes(end+1) = img_size;
        results.manual_times(end+1) = t_manual;
        results.builtin_times(end+1) = t_builtin;
        results.speedups(end+1) = speedup;
        results.matches(end+1) = is_correct;

        % Print results
        fprintf('%-20s | %-12.4f | %-12.4f | %-8.2fx | %-8s\n', img_list{k}, t_manual, t_builtin, speedup, string(is_correct));

        % Save edge detection result images
        save_edge_images(out_manual, img_list{k}, img_folder);
    end

    fprintf('%s\n', repmat('-', 1, 70));
    fprintf('Average speedup: %.2fx\n\n', mean(results.speedups));

    % Generate performance plots
    plot_results(results);

    % Display edge detection results (original vs edges)
    display_edge_results(img_folder, img_list);

    % Save results to file
    save('convolution_results.mat', 'results');
    fprintf('Results saved to convolution_results.mat\n');
end

function save_edge_images(edge_output, img_name, img_folder)
    % Save edge detection result as image
    output_folder = fullfile(img_folder, 'results');
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Normalize to 0-255 range for saving
    edge_normalized = uint8(255 * edge_output / max(edge_output(:)));

    % Create output filename
    [~, name, ~] = fileparts(img_name);
    output_path = fullfile(output_folder, [name '_edges.png']);

    imwrite(edge_normalized, output_path);
    fprintf('  -> Saved edge result: %s\n', output_path);
end

function display_edge_results(img_folder, img_list)
    % Display original images alongside their edge detection results
    num_images = length(img_list);
    figure('Name', 'Sobel Edge Detection Results', 'Position', [50 50 1400 800]);

    for k = 1:num_images
        img_path = fullfile(img_folder, img_list{k});
        if ~exist(img_path, 'file')
            continue;
        end

        % Load original image
        original = imread(img_path);
        if size(original, 3) == 3
            original = rgb2gray(original);
        end

        % Load edge detection result
        [~, name, ~] = fileparts(img_list{k});
        edge_path = fullfile(img_folder, 'results', [name '_edges.png']);
        if ~exist(edge_path, 'file')
            continue;
        end
        edges = imread(edge_path);

        % Display original
        subplot(2, num_images, k);
        imshow(original);
        title(['Original: ' name], 'Interpreter', 'none', 'FontSize', 8);

        % Display edges
        subplot(2, num_images, num_images + k);
        imshow(edges);
        title('Sobel Edges', 'FontSize', 8);
    end

    sgtitle('Sobel Edge Detection: Original vs Edge-Detected Images');
    saveas(gcf, 'edge_detection_results.png');
    fprintf('Edge detection comparison saved to edge_detection_results.png\n');
end

function plot_results(results)
    % Create performance comparison plots
    figure('Name', 'Convolution Performance Analysis', 'Position', [100 100 1200 500]);

    % Plot 1: Execution times comparison
    subplot(1, 3, 1);
    x = 1:length(results.image_sizes);
    bar(x, [results.manual_times' results.builtin_times']);
    set(gca, 'XTickLabel', results.image_sizes);
    xlabel('Image Size (pixels)');
    ylabel('Execution Time (seconds)');
    title('Execution Time Comparison');
    legend('Manual', 'Built-in', 'Location', 'northwest');
    grid on;

    % Plot 2: Speedup vs Image Size
    subplot(1, 3, 2);
    plot(results.image_sizes, results.speedups, 'b-o', 'LineWidth', 2, 'MarkerSize', 8);
    xlabel('Image Size (pixels)');
    ylabel('Speedup (Manual / Built-in)');
    title('Speedup vs Image Size');
    grid on;

    % Plot 3: Log-scale timing comparison
    subplot(1, 3, 3);
    loglog(results.image_sizes, results.manual_times, 'r-o', 'LineWidth', 2, 'MarkerSize', 8);
    hold on;
    loglog(results.image_sizes, results.builtin_times, 'b-s', 'LineWidth', 2, 'MarkerSize', 8);
    hold off;
    xlabel('Image Size (pixels)');
    ylabel('Execution Time (seconds)');
    title('Timing (Log-Log Scale)');
    legend('Manual', 'Built-in', 'Location', 'northwest');
    grid on;

    % Save figure
    saveas(gcf, 'performance_analysis.png');
    fprintf('Performance plot saved to performance_analysis.png\n');
end

%% ========================================================================
%  PART 1: Manual 2D Convolution Implementation
%  ========================================================================
function output = my_conv2(img_path)
    % Manual 2D convolution using nested loops
    % Implements Sobel edge detection without using conv2() or imfilter()
    % Uses zero-padding to maintain output size equal to input size ('same' mode)

    img = imread(img_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img);
    [rows, cols] = size(img);

    % Sobel kernels for horizontal and vertical edge detection
    Gx_k = [-1 0 1; -2 0 2; -1 0 1];  % Horizontal edges
    Gy_k = [-1 -2 -1; 0 0 0; 1 2 1];  % Vertical edges

    % Initialize output matrices
    Gx = zeros(rows, cols);
    Gy = zeros(rows, cols);

    % Zero-padding: add 1 pixel border to handle boundaries
    % This matches conv2(..., 'same') behavior
    img_p = padarray(img, [1 1], 0);

    % Manual convolution using nested loops
    for i = 1:rows
        for j = 1:cols
            % Extract 3x3 region centered at current pixel
            region = img_p(i:i+2, j:j+2);
            % Element-wise multiply and sum (convolution operation)
            Gx(i, j) = sum(sum(region .* Gx_k));
            Gy(i, j) = sum(sum(region .* Gy_k));
        end
    end

    % Compute gradient magnitude using absolute value approximation
    % This avoids expensive sqrt() operation: |Gx| + |Gy| instead of sqrt(Gx^2 + Gy^2)
    output = abs(Gx) + abs(Gy);
end

%% ========================================================================
%  PART 2: Built-in 2D Convolution Implementation
%  ========================================================================
function output = inbuilt_conv2(img_path)
    % Uses MATLAB's conv2() function for 2D convolution
    % Note: conv2 performs convolution (flips kernel), so we use rot180 kernels
    % to match the manual correlation-style implementation
    img = imread(img_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = double(img);

    % Sobel kernels - same as manual implementation
    Gx_k = [-1 0 1; -2 0 2; -1 0 1];
    Gy_k = [-1 -2 -1;  0 0 0;  1 2 1];

    % conv2 flips the kernel (true convolution), but Sobel kernels are
    % symmetric under 180° rotation, so output matches manual implementation
    % Using 'same' mode to match the zero-padded manual output size
    Gx = conv2(img, Gx_k, 'same');
    Gy = conv2(img, Gy_k, 'same');

    % Compute gradient magnitude using absolute value approximation
    % This avoids expensive sqrt() operation: |Gx| + |Gy| instead of sqrt(Gx^2 + Gy^2)
    output = abs(Gx) + abs(Gy);
end
