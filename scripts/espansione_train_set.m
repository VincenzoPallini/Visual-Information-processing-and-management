%% Creazione New Labels Ridotto
csvPath = './../csv/train_set.csv';
outPath = './../csv/train_set_ridotto.csv';

data = readtable(csvPath);
outFile = fopen(outPath, 'w');
imageNames = data{:, 1};
classes = data{:, 2};

% Numero massimo di immagini da tenere per classe
numToKeep = 180;

uniqueClasses = unique(classes);

for i = 1:length(uniqueClasses)
    className = uniqueClasses(i);
    classIndices = (classes == className);
    classImages = imageNames(classIndices);

    for j = 1:min(numToKeep, length(classImages))
        fprintf(outFile, '%s,%i\n', classImages{j}, className);
    end
end

disp(['File filtrato salvato come: ', outPath]);

%% Rimozione file labeled dal train_unlabeled
newLabeldata = readtable('../csv/train_small_pulito_espanso.csv');
unlabeledData = readtable('../csv/train_unlabeled.csv');

newLabel_imageNames = newLabeldata{:, 1};
unlabeled_imageNames = unlabeledData{:, 1};

[~, i] = intersect(newLabel_imageNames, unlabeled_imageNames);
unlabeledData(i, :) = [];

outputFile = '../csv/train_unlabeled.csv';
writetable(unlabeledData, outputFile);

disp(['File filtrato salvato come: ', outputFile]);

%% Unione dei train set piccoli
newLabeldata = readtable('../csv/new_labels.csv');
smallData = readtable('../csv/train_set.csv');

result = [newLabeldata; smallData];

outputFile = '../csv/train_set.csv';
writetable(result, outputFile);

disp(['File filtrato salvato come: ', outputFile]);

%% Rimozione Duplicati
trainSet = readtable('../csv/train_set.csv');

uniqueData = unique(trainSet, 'rows');

outputFile = '../csv/train_set.csv';
writetable(uniqueData, outputFile);

disp(['File filtrato salvato come: ', outputFile]);
