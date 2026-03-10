%% ========================================================================
%  PART 1: Mandelbrot Set Image Plotting and Saving
%  ========================================================================
%
% TODO: Implement Mandelbrot set plotting and saving function
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