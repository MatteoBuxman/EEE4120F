function mandelbrot_plot(image_data, resolution_name, save_path)

    [dir_path, ~, ~] = fileparts(save_path);
    if ~isempty(dir_path) && ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end

    figure('Visible', 'off');
    imagesc(image_data);
    colormap(hot);
    colorbar;
    axis image off;
    title(sprintf('Mandelbrot Set — %s', resolution_name),'FontSize', 12, 'FontWeight', 'bold');
    saveas(gcf, save_path);
    close(gcf);

    fprintf('Saved Mandelbrot image: %s\n', save_path);
end