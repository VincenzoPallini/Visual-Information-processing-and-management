%% ADDESTRAMENTO DEL MODELLO
imdsTrain = loadSet('train');
if exist('../saved_files/trainedNet.mat', 'file')
    load('../saved_files/trainedNet.mat', 'trainedNet');
    disp('Rete preaddestrata caricata da file pre-esistente: trainedNet.mat');
else
    featuresTrain = loadFeatures('train');
    disp('Addestramento rete in corso...');
    trueLabels = imdsTrain.Labels;
    trainedNet = fitcecoc(featuresTrain, trueLabels);
    save('../saved_files/trainedNet.mat', 'trainedNet');
    disp('Rete addestrata salvata in ../saved_files/trainedNet.mat');
end
%% Accuracy su Test Set
imdsVal = loadSet('validation');
featuresTest = loadFeatures('validation');

disp('Calcolo accuracy su validation test set in corso...');
[predictedLabels, scores] = predict(trainedNet, featuresTest);
trueLabels = imdsVal.Labels;
accuracy = mean(predictedLabels == trueLabels);

disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);