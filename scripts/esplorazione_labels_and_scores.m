%% Load Labels and Scores
if exist('../saved_files/labels_and_score.mat', 'file')
    load('../saved_files/labels_and_score.mat', 'predictedLabels', 'scores');
    disp('Labels e scores caricati da file pre-esistente: ../saved_files/labels_and_score.mat');
end

%% Istogramma Scores Generale
figure(1)
histogram(scores, 10);
title("Distribuzione Scores con Resnet18")

%% Istogramma Scores per Classe
figure(2)
histogram(predictedLabels)
title("Distribuzione Labels con Resnet18")

%% Media Scores
predictedLabels = categorical(predictedLabels);
averageScores = groupsummary(scores, predictedLabels, 'mean');
uniqueLabels = unique(predictedLabels);
meanScores = averageScores; 

figure(3)
bar(uniqueLabels, meanScores);
xlabel('Labels');
ylabel('Score Medio');
title('Score Medio per Label');