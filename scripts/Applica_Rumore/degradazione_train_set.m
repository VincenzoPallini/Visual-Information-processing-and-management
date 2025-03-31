function degradazione_train_set()
    base_path = '..\';
    input_path = fullfile(base_path, 'image_sets', 'train_set');
    output_path = fullfile(base_path, 'image_sets', 'train_set_degraded');
    
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % Parametri per le degradazioni
    quality = 9;          % per compressione JPEG
    sigma_value = 3;      % per filtro gaussiano
    gauss_var = 0.06;     % per rumore gaussiano
    
    data = readtable('..\csv\train_small_pulito_espanso.csv');
    
    if iscell(data.Var2)
        data.Var2 = cell2mat(data.Var2);
    end
    
    classes = unique(data.Var2);
    
    for class_idx = 1:length(classes)
        current_class = classes(class_idx);
        
        class_images = data(data.Var2 == current_class, :);
        num_images = height(class_images);
        
        rng(42); 
        shuffled_indices = randperm(num_images);
        

        group_size = floor(num_images / 4);
        
        idx_start = 1;
        idx_end = group_size;
        group1 = shuffled_indices(idx_start:idx_end);  % pulito
        
        idx_start = idx_end + 1;
        idx_end = idx_end + group_size;
        group2 = shuffled_indices(idx_start:idx_end);  % compressione
        
        idx_start = idx_end + 1;
        idx_end = idx_end + group_size;
        group3 = shuffled_indices(idx_start:idx_end);  % filtro gaussiano
        
        idx_start = idx_end + 1;
        idx_end = num_images;
        group4 = shuffled_indices(idx_start:idx_end);  % rumore gaussiano
        
        %% Gruppo 1: Pulito (copia immagine senza degradazione)
        for i = 1:length(group1)
            img_name = class_images.Var1{group1(i)};
            img_path = fullfile(input_path, img_name);
            output_img_path = fullfile(output_path, img_name);

            try
                img = imread(img_path);
                imwrite(img, output_img_path);
            catch e
                fprintf('Errore nel processare %s: %s\n', img_name, e.message);
                continue;
            end
        end
        
        %% Gruppo 2: Compressione
        for i = 1:length(group2)
            img_name = class_images.Var1{group2(i)};
            img_path = fullfile(input_path, img_name);
            output_img_path = fullfile(output_path, img_name);
            
            try
                img = imread(img_path);
                
                imwrite(img, output_img_path, 'jpg', 'Quality', quality);
                
            catch e
                fprintf('Errore nel processare %s: %s\n', img_name, e.message);
                continue;
            end
        end
        
        %% Gruppo 3: Filtro Gaussiano
        for i = 1:length(group3)
            img_name = class_images.Var1{group3(i)};
            img_path = fullfile(input_path, img_name);
            output_img_path = fullfile(output_path, img_name);
            
            try
                img = im2double(imread(img_path));
                
                filtered_img = applica_filtro_gaussiano(img, sigma_value);
                
                imwrite(filtered_img, output_img_path);
                
            catch e
                fprintf('Errore nel processare %s: %s\n', img_name, e.message);
                continue;
            end
        end
        
        %% Gruppo 4: Rumore Gaussiano
        for i = 1:length(group4)
            img_name = class_images.Var1{group4(i)};
            img_path = fullfile(input_path, img_name);
            output_img_path = fullfile(output_path, img_name);
            
            try
                img = im2double(imread(img_path));
                
                noisy_img = applica_rumore_gaussiano(img, gauss_var);
                
                imwrite(noisy_img, output_img_path);
                
            catch e
                fprintf('Errore nel processare %s: %s\n', img_name, e.message);
                continue;
            end
        end
        
        fprintf('Completata classe %d/%d\n', class_idx, length(classes));
    end
    fprintf('Processo completato!\n');
end


function [im] = applica_filtro_gaussiano(im,sigma_value)
    
    im = imgaussfilt(im,sigma_value);

end

function [im] = applica_rumore_gaussiano(im,gauss_var)
    
    im = imnoise(im,'gaussian',0.0,gauss_var);

end