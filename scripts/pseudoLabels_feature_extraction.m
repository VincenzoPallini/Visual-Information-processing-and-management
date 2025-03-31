%% Creazione Labels and Scores
if exist('../saved_files/labels_and_score.mat', 'file')
    load('../saved_files/labels_and_score.mat', 'predictedLabels', 'scores');
    disp('Labels e scores caricati da file pre-esistente: ../saved_files/labels_and_score.mat');
else
    if exist('../saved_files/trainedNet.mat', 'file')
        load('../saved_files/trainedNet.mat', 'trainedNet');
        disp('Rete preaddestrata caricata da file pre-esistente: ../saved_files/trainedNet.mat');

        featureUnlabeled = loadFeatures('unlabeled');

        disp('Creazione labels in corso...');
        [predictedLabels, scores] = predict(trainedNet, featureUnlabeled);
        save('../saved_files/labels_and_score.mat', 'predictedLabels', 'scores');
        disp('Labels e scores salvati su file in saved_files/labels_and_score.mat.');
    else
    disp('Errore! Non esiste nessuna rete addestrata')
    end
end