function demo_filtro_gaussiano()
    original_path = '..\image_sets\val_set\val_000473.jpg';
    degraded_path = '..\image_sets\val_set_degraded\val_000473.jpg';
    
    original_image = im2double(imread(original_path));
    degraded_image = im2double(imread(degraded_path));
    
    sigma_value = 3; 
    filtered_image = applica_filtro_gaussiano(original_image, sigma_value);
    
    figure;
    subplot(1,3,1);
    imshow(original_image);
    title('Immagine Originale');
    
    subplot(1,3,2);
    imshow(filtered_image);
    title(['Immagine Filtrata (sigma = ' num2str(sigma_value) ')']);
    
    subplot(1,3,3);
    imshow(degraded_image);
    title('Immagine Degradata');
    

end

function filtered_image = applica_filtro_gaussiano(input_image, sigma)
    kernel_size = 2 * ceil(3 * sigma) + 1; 
    gaussian_kernel = fspecial('gaussian', kernel_size, sigma);
    
    filtered_image = imfilter(input_image, gaussian_kernel, 'symmetric');
end
