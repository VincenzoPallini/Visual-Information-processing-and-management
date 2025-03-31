%% CARICAMENTO DEI SETS
imdsTrain = loadSet('train');
imdsVal = loadSet('validation');
[classCounts, classNames] = groupcounts(imdsTrain.Labels);

augmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-15, 15], ...
    'RandYTranslation', [-15, 15], ...
    'RandScale', [0.8, 1.2] ...
);

%% Creazione Augmented Dataset

if exist('../saved_files/augTrain.mat', 'file')
    load('../saved_files/augTrain.mat', 'augTrain');
    disp(length(augTrain.Files))
    disp('Augmented dataset caricato da file pre-esistente: saved_files/augTrain.mat');
else
    disp('Creazione augmented dataset in corso...');
    [~, classNames] = groupcounts(imdsTrain.Labels);
    uniqueClasses = unique(classNames);
    
    augmentedDir = './../image_sets/augmented_set';
    
    augmentedFiles = {};
    augmentedLabels = {};
    % numToAugment = 0;
    
    for i = 1:length(uniqueClasses)
        class = uniqueClasses(i);
        classIndex = (imdsTrain.Labels == class);
        files = imdsTrain.Files(classIndex);
        numToAugment = max(classCounts) - sum(classIndex);

        for j = 1:numToAugment
            img = imread(files{mod(j-1, numel(files)) + 1});
            imgAug = augment(augmenter, img);
            augmentedFiles = [augmentedFiles; fullfile(augmentedDir, ['augmented_' num2str(i) '_' num2str(j) '.jpg'])];
            imwrite(imgAug, augmentedFiles{end});
            augmentedLabels = [augmentedLabels; class];
        end
    end
    
    % Combina i dataset
    augTrain = imageDatastore([imdsTrain.Files; augmentedFiles], ...
        'Labels', [imdsTrain.Labels; categorical(augmentedLabels)]);
    
    [classCounts, classNames] = groupcounts(augTrain.Labels);
    
    bar(classNames, classCounts);
    save('../saved_files/augTrain.mat', 'augTrain');
    disp('Augmented dataset salvato su file in saved_files/augTrain.mat.');
end

%% ADDESTRAMENTO DEL MODELLO
numClasses = 251;
netName = "resnet101";  % resnet18 / resnet50 / resnet101 / mobilenetV2
net = imagePretrainedNetwork(netName, NumClasses = numClasses);

% analyzeNetwork(net)
% pause
   
inputImageSize  = net.Layers(1).InputSize;

% Augmented Datasets
augimdsTrain = augmentedImageDatastore(inputImageSize(1:2), augTrain, 'DataAugmentation', augmenter);
augimdsVal = augmentedImageDatastore(inputImageSize(1:2), imdsVal);

if exist('../saved_files/trainedNet.mat', 'file')
    load('../saved_files/trainedNet.mat', 'trainedNet');
    disp('Rete preaddestrata caricata da file pre-esistente: ../saved_files/trainedNet.mat');
else
    disp('Addestramento rete in corso...');
    
    lgraph = layerGraph(net);
    
    % Freeze dei layer con pesi addestrabili
    N = 301; % resnet18 = 19/29/35, mobilenetv2 = 151, resnet101 = 301, resnet50 = 141
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    for i = 1:N
        layer = layers(i);
        if isprop(layer, 'WeightLearnRateFactor') && isprop(layer, 'BiasLearnRateFactor')
            layer.WeightLearnRateFactor = 0;
            layer.BiasLearnRateFactor = 0;
            layers(i) = layer;
        end
    end

    lgraph = createGraph(layers, connections);
    
    newLayers = [
        dropoutLayer('Name', 'dropout')
    ];
    
    % Rimuove i vecchi livelli e aggiunge quelli nuovi

    % Resnet 18/101
    lgraph = disconnectLayers(lgraph, 'pool5', 'fc1000');
    lgraph = addLayers(lgraph, newLayers(1));
    lgraph = connectLayers(lgraph, 'pool5', 'dropout');
    lgraph = connectLayers(lgraph, 'dropout', 'fc1000');

    % % MobilenetV2
    % lgraph = disconnectLayers(lgraph, 'global_average_pooling2d_1', 'Logits');
    % lgraph = addLayers(lgraph, newLayers(1));
    % lgraph = connectLayers(lgraph, 'global_average_pooling2d_1', 'dropout');
    % lgraph = connectLayers(lgraph, 'dropout', 'Logits');

    % % Resnet50
    % lgraph = disconnectLayers(lgraph, 'avg_pool', 'fc1000');
    % lgraph = addLayers(lgraph, newLayers(1));
    % lgraph = connectLayers(lgraph, 'avg_pool', 'dropout');
    % lgraph = connectLayers(lgraph, 'dropout', 'fc1000');

    net = dlnetwork(lgraph);
    % analyzeNetwork(net);
         
    % Impostazioni Fine Tuning
    options = trainingOptions("adam", ...
        ValidationData = augimdsVal, ...
        ValidationFrequency = 10, ...
        ValidationPatience = 5, ...
        MiniBatchSize = 64, ...
        MaxEpochs = 50, ...
        InitialLearnRate = 5e-5, ...
        LearnRateDropPeriod = 10, ...
        LearnRateDropFactor = 0.1, ...
        Plots = "training-progress", ...
        Metrics = "accuracy", ...
        Verbose=false);
    
    % Addestramento
    trainedNet = trainnet(augimdsTrain, net, "crossentropy", options);
    
    save('../saved_files/trainedNet.mat', 'trainedNet');
    disp('Rete addestrata salvata in ../saved_files/trainedNet.mat');
end
%% Accuracy su Test Set
disp('Calcolo accuracy su validation test in corso...');
scores = minibatchpredict(trainedNet, augimdsVal);
[predictedLabels, ~] = scores2label(scores, classNames);
trueLabels = imdsVal.Labels;
acc = mean(trueLabels == predictedLabels);

disp(['Test Accuracy: ', num2str(acc * 100), '%']);

uniqueClasses = categories(trueLabels);
classAccuracies = zeros(numel(uniqueClasses), 1);

for i = 1:numel(uniqueClasses)
    currentClass = uniqueClasses{i};
    idx = trueLabels == currentClass;
    classAccuracies(i) = mean(predictedLabels(idx) == trueLabels(idx));
end

% Accuracy per classe
figure;
bar(uniqueClasses, classAccuracies * 100);
xlabel('Classi');
ylabel('Accuracy (%)');
title('Accuracy per Classe');