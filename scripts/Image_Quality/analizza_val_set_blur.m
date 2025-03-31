function analizza_val_set_blur()
    folder_path = '..\image_sets\val_set_degraded';
    
    image_files  = dir(fullfile(folder_path, '*.png'));
    image_files2 = dir(fullfile(folder_path, '*.jpg'));
    image_files3 = dir(fullfile(folder_path, '*.jpeg'));
    
    image_files = [image_files; image_files2; image_files3];
    
    nImages = length(image_files);
    
    all_blur_scores = zeros(nImages, 1);
    
    blurry_images = struct('name', {}, 'blur_score', {});
    
    blur_threshold = 100; 
    
    for i = 1:nImages
        try
            current_filename = fullfile(folder_path, image_files(i).name);
            img = imread(current_filename);
            
            blur_score = BlurDetect(img);
            
            all_blur_scores(i) = blur_score;
            
            if blur_score < blur_threshold
                blurry_images(end+1).name       = image_files(i).name; 
                blurry_images(end).blur_score   = blur_score;
            end
            
            fprintf('Immagine: %s, blur score = %.3f\n', ...
                    image_files(i).name, blur_score);
            
        catch ME
            fprintf('Error processing image %s: %s\n', ...
                    image_files(i).name, ME.message);
            continue;
        end
    end
    
    valid_scores = all_blur_scores(all_blur_scores ~= 0);
    if ~isempty(valid_scores)
        fprintf('\nStatistiche Blur:\n');
        fprintf('Media blur score: %.3f\n', mean(valid_scores));
        fprintf('Deviazione standard: %.3f\n', std(valid_scores));
        fprintf('Minimo: %.3f\n', min(valid_scores));
        fprintf('Massimo: %.3f\n', max(valid_scores));
    end
    
    if ~isempty(blurry_images)
        blur_scores = [blurry_images.blur_score];
        [~, idx]    = sort(blur_scores, 'ascend');  
        blurry_images = blurry_images(idx);
        
        fprintf('\nElenco delle immagini sfocate (ordinate per livello di blur):\n');
        fprintf('------------------------------------------------\n');
        for i = 1:length(blurry_images)
            fprintf('%-40s\t%.3f\n', ...
                    blurry_images(i).name, blurry_images(i).blur_score);
        end
        fprintf('------------------------------------------------\n');
        fprintf('Totale immagini sfocate: %d\n', length(blurry_images));
    else
        fprintf('\nNessuna immagine sfocata trovata\n');
    end
end
