function NoiseLevel_val_set_degraded()
    folder_path = '..\image_sets\val_set_degraded';
    
    image_files = dir(fullfile(folder_path, '*.png'));
    image_files2 = dir(fullfile(folder_path, '*.jpg'));
    image_files3 = dir(fullfile(folder_path, '*.jpeg'));
    
    image_files = [image_files; image_files2; image_files3];
    
    nImages = length(image_files);
    
    num_noisy_images = 0;
    
    threshold = 10;
    
    all_noise_levels = zeros(nImages,1);
    
    noisy_images = struct('name', {}, 'noise_level', {});
    
    patchsize = 7;
    decim = 0;
    conf = 1-1E-6;
    itr = 3;
    
    for i = 1:nImages
        try
            current_filename = fullfile(folder_path, image_files(i).name);
            img = imread(current_filename);
            
            img = double(img);
            
            [nlevel, ~, ~] = NoiseLevel(img, patchsize, decim, conf, itr);
            
            noise_value = mean(nlevel);
            
            all_noise_levels(i) = noise_value;
            
            if noise_value > threshold
                num_noisy_images = num_noisy_images + 1;
            end
            
            if noise_value > 2.0
                noisy_images(end+1).name = image_files(i).name;
                noisy_images(end).noise_level = noise_value;
            end
            
            fprintf('Immagine: %s, rumore stimato = %.3f\n', image_files(i).name, noise_value);
        catch ME
            fprintf('Error processing image %s: %s\n', image_files(i).name, ME.message);
            continue;
        end
    end
    
    fprintf('\nNumero di immagini rumorose: %d su %d\n', num_noisy_images, nImages);
    
    valid_measurements = all_noise_levels(all_noise_levels ~= 0);
    if ~isempty(valid_measurements)
        fprintf('Media livello rumore: %.3f\n', mean(valid_measurements));
        fprintf('Deviazione standard: %.3f\n', std(valid_measurements));
        fprintf('Minimo: %.3f\n', min(valid_measurements));
        fprintf('Massimo: %.3f\n', max(valid_measurements));
    end
    
    if ~isempty(noisy_images)
        noise_levels = [noisy_images.noise_level];
        [~, idx] = sort(noise_levels, 'descend');
        noisy_images = noisy_images(idx);
        
        fprintf('\nElenco delle immagini con rumore > 2.0 (ordinate per livello di rumore):\n');
        fprintf('------------------------------------------------\n');
        for i = 1:length(noisy_images)
            fprintf('%-40s\t%.3f\n', noisy_images(i).name, noisy_images(i).noise_level);
        end
        fprintf('------------------------------------------------\n');
        fprintf('Totale immagini con rumore > 2.0: %d\n', length(noisy_images));
    else
        fprintf('\nNessuna immagine trovata con rumore > 2.0\n');
    end
end