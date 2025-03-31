%% CARICAMENTO DEI SETS E DEL MODELLO
imdsTrain = loadSet('train');
imdsUnlabeled = loadSet('unlabeled');

[classCounts, classNames] = groupcounts(imdsTrain.Labels);

%% Creazione Labels and Scores
if exist('../saved_files/labels_and_score.mat', 'file')
    load('../saved_files/labels_and_score.mat', 'predictedLabels', 'scores');
    disp('Labels e scores caricati da file pre-esistente: ../saved_files/labels_and_score.mat');
else
    if exist('../saved_files/trainedNet.mat', 'file')
        load('../saved_files/trainedNet.mat', 'trainedNet');
        disp('Rete preaddestrata caricata da file pre-esistente: ../saved_files/trainedNet.mat');

        inputImageSize  = trainedNet.Layers(1).InputSize;
    
        % Nuove Labels
        augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);

        disp('Creazione labels in corso...');
        scores = minibatchpredict(trainedNet, augimdsUnlabeled);
        [predictedLabels, scores] = scores2label(scores, classNames);
        save('../saved_files/labels_and_score.mat', 'predictedLabels', 'scores');
        disp('Labels e scores salvati su file in saved_files/labels_and_score.mat.');
    else
    disp('Errore! Non esiste nessuna rete addestrata')
    end
end

%% Creazione file csv
if exist('../csv/new_labels.csv', 'file')
    disp('File ../csv/new_labels.csv già esistente')
else
    disp('Creazione file ../csv/new_labels.csv')
    newLabelCSV = '../csv/new_labels.csv';
    csvFile = fopen(newLabelCSV, 'w');
    
    if exist('../saved_files/trainedNet.mat', 'file')
        load('../saved_files/trainedNet.mat', 'trainedNet');
        disp('Rete preaddestrata caricata da file pre-esistente: ../saved_files/trainedNet.mat');

        inputImageSize  = trainedNet.Layers(1).InputSize;
        augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);
    end

    imagePath = augimdsUnlabeled.Files;

    threshold = 0.00;

    for i = 1:length(imagePath)
        [filepath, name, ext] = fileparts(imagePath{i});
        if(scores(i) >= threshold)
            fprintf(csvFile, '%s.jpg,%s\n', name, predictedLabels(i));
        end
    end
    disp('Nuove Lables salvate in ../csv/new_labels.csv');
    fclose(csvFile);
end