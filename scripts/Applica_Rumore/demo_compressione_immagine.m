function demo_compressione_immagine()
    original_path = '..\image_sets\val_set\val_000013.jpg';
    degraded_path = '..\image_sets\val_set_degraded\val_000013.jpg';
    
    original_image = imread(original_path);
    degraded_image = imread(degraded_path);
    
    quality = 9; % (range: 1-100)
    temp_file = 'temp_compressed.jpg';
    imwrite(original_image, temp_file, 'jpg', 'Quality', quality);
    compressed_image = imread(temp_file);
    
    figure;
    
    subplot(1,3,1);
    imshow(original_image);
    title('Immagine Originale');
    
    subplot(1,3,2);
    imshow(compressed_image);
    title(['Immagine Compressa (Quality = ' num2str(quality) ')']);
    
    subplot(1,3,3);
    imshow(degraded_image);
    title('Immagine Degradata');
    
    info_orig = imfinfo(original_path);
    info_comp = imfinfo(temp_file);
    info_degr = imfinfo(degraded_path);
    
    compression_ratio = info_orig.FileSize / info_comp.FileSize;
    degradation_ratio = info_orig.FileSize / info_degr.FileSize;
    
    fprintf('Rapporto di compressione (Originale vs Compressa): %.2f:1\n', compression_ratio);
    fprintf('Rapporto di compressione (Originale vs Degradata): %.2f:1\n', degradation_ratio);
    
    delete(temp_file);
end