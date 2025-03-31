% Inizializzazione
datasetPath = './train_set';
imageFiles = dir(fullfile(datasetPath, '*.jpg'));
numImages = length(imageFiles);

net = resnet18;
% analyzeNetwork(net)
featureSize = 512;
imageSize = [224, 224];

numClusters = 251;

%% Estrazione delle feature (30 minuti circa)
featureLayer = 'pool5'; % Scelta perchè vicino all'output e non troppo dettagliata (per performance)

disp("Estrazione Feature dalle immagini in corso...")
if isfile('features.mat')
    disp('Caricamento delle feature dal file salvato...');
    load('features.mat', 'features');
else
    tic;
    features = zeros(numImages, featureSize);

    parfor i = 1:numImages
        disp(i)
        imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
        img = imread(imgPath);
        img = gpuArray(img);
        
        % grayscale in RGB
        img = imresize(img, imageSize);
        if size(img, 3) ~= 3
            img = repmat(img, 1, 1, 3); 
        end
        
        featureVector = activations(net, img, featureLayer, 'OutputAs', 'rows');
        features(i, :) = featureVector;
    end
    toc;
    save('features.mat', 'features');
end

%% Studio Componenti Principali
% [coeff, score, latent] = pca(featuresNormalized);
% 
% explainedVariance = latent / sum(latent) * 100;
% cumulativeVariance = cumsum(explainedVariance);
% sumTo90 = find(cumulativeVariance >= 90, 1);
% disp(numComponents90);
%
% % Grafico
% figure;
% bar(explainedVariance(1:featureSize)); 
% title('Scree Plot');
% xlabel('Componente Principale');
% ylabel('Varianza (%)');

%% Creazione Vocabolario
disp('Clustering con k-means...');

if isfile('kmeans.mat')
    disp('Caricamento degli ID dal file salvato...');
    load('kmeans.mat', 'clusterIdx');
else
    sumTo90 = 205; % Trovato da "Studio Componenti Principali"
    fprintf('Riduzione dimensionale con PCA...\n');
    features = normalize(features);
    [coeff, features] = pca(features, 'NumComponents', sumTo90);

    
    [clusterIdx, ~] = kmeans(features, numClusters);
    save('kmeans.mat', 'clusterIdx');
end

%% Analisi Empirica dei Cluster
% disp("Mostro 25 immagini per cluster...")
% for k = 1:numel(numClusters)
%     fprintf('Immagini nel cluster %d:\n', k);
%     clusterImages = find(clusterIdx == k);
%     for j = 1:min(25, numel(clusterImages))
%         imgPath = fullfile(imageFiles(clusterImages(j)).folder, imageFiles(clusterImages(j)).name);
%         img = imread(imgPath);
%         subplot(5, 5, j);
%         imshow(img);
%         title(sprintf('Cluster %d', k));
%     end
%     pause;
% end

%% Rimozione Immagini di Non Cibo
disp("Spostando i file di non cibo...")
% Cluster identificati come "non cibo" rilevati da "Analisi Empirica dei Cluster"
nonCiboClusters = [4, 15, 37, 40, 42, 48, 54, 91, 96, 104, 106, ... 
     107, 113, 125, 126, 136, 138, 158, 164, 166, 174, 178, 191, 218, ... 
     223, 204, 235, 236, 249]; 

% Cartella di destinazione del "non cibo"
nonCiboPath = './train_set/not_food';
if ~exist(nonCiboPath, 'dir')
    mkdir(nonCiboPath);
end

for k = nonCiboClusters
    clusterImages = find(clusterIdx == k);
    for j = 1:numel(clusterImages)
    imgPath = fullfile(imageFiles(clusterImages(j)).folder, imageFiles(clusterImages(j)).name);
    [~, fileName, fileExt] = fileparts(imgPath); 
    destFile = fullfile(nonCiboPath, [fileName, fileExt]);
    movefile(imgPath, destFile);
    end
end
disp("Spostamento completato!")