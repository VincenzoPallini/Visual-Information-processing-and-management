img_path = '..\..\image_sets\val_set_degraded\val_000005.jpg';
test_img = imread(img_path);

test_img = double(test_img);

% Parametri di default
patchsize = 7; 
decim = 0;      
conf = 1-1E-6;  
itr = 3;    

[nlevel, th, num] = NoiseLevel(test_img, patchsize, decim, conf, itr);

fprintf('Immagine analizzata: %s\n', img_path);
if length(nlevel) > 1
    fprintf('Livello di rumore stimato per canale:\n');
    for i = 1:length(nlevel)
        fprintf('Canale %d: %.4f\n', i, nlevel(i));
    end
    fprintf('Media dei livelli di rumore: %.4f\n', mean(nlevel));
else
    fprintf('Livello di rumore stimato: %.4f\n', nlevel);
end
fprintf('Valore soglia: %.4f\n', th);
fprintf('Numero di patch utilizzate: %d\n', num);

figure(1);
if size(test_img, 3) == 1
    imshow(test_img, []);
else
    imshow(test_img ./ 255);
end
title('Immagine Originale');