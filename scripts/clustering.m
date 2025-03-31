%% ESTRAZIONE FEATURES (5 minuti)
imdsUnlabeled = loadSet('unlabeled');
numClasses = 251;
net = resnet101;  % resnet18 / resnet50 / resnet101
layer = 'pool5';

% analyzeNetwork(net)

% Augmented Datasets
inputImageSize  = net.Layers(1).InputSize;

augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);

if exist('../saved_files/feature_unlabeled_resnet101.mat', 'file')
    load('../saved_files/feature_unlabeled_resnet101.mat', 'features');
    disp('Feature caricate da file pre-esistente: feature_unlabeled_resnet101.mat');
else
    disp('Estrazione Features in corso...');
    tic
    features = activations(net,augimdsUnlabeled,layer,'OutputAs','rows');
    toc
    save('../saved_files/feature_unlabeled_resnet101.mat', 'features');
    disp('Rete addestrata salvata in ../saved_files/trainedNet.mat');
end

%% Studio Componenti Principali
% features = zscore(features);
% [coeff, score, latent] = pca(features);
% 
% explainedVariance = latent / sum(latent) * 100;
% cumulativeVariance = cumsum(explainedVariance);
% sumTo90 = find(cumulativeVariance >= 90, 1);
% disp(sumTo90);

%% Creazione Vocabolario
disp('Clustering con k-means...');

if isfile('unlabeled_kmeans.mat')
    disp('Caricamento degli ID dal file salvato...');
    load('unlabeled_kmeans.mat', 'clusterIdx', 'centroids');
else
    % sumTo90 = 205; % Trovato da "Studio Componenti Principali"
    fprintf('Creazione Vocabolario in corso...\n');
    
    [coeff, features] = pca(features);
    % , 'NumComponents', sumTo90
    
    tic
    [clusterIdx, centroids] = kmeans(features, numClasses, 'MaxIter', 1000, 'Replicates', 5 );
    toc
    save('unlabeled_kmeans.mat', 'clusterIdx', 'centroids');
end

%% Analisi Empirica dei Cluster
disp("Mostro 25 immagini per cluster...")
imageFiles = imdsUnlabeled.Files;
for k = 1:numClasses
    fprintf('Immagini nel cluster %d:\n', k);
    clusterImages = find(clusterIdx == k);
    for j = 1:min(25, numel(clusterImages))
        imgPath = fullfile(imageFiles{clusterImages(j)});
        img = imread(imgPath);
        subplot(5, 5, j);
        imshow(img);
        title(sprintf('Cluster %d', k));
    end
    pause;
end

