%% Creazione cartella da csv con label
% Lo script genera una cartella con il nome fornito, tante sottocartelle
% quante le classi e smista le immagini per classe creandone una COPIA

csvPath = './csv/train_small_pulito.csv';
datasetPath = './train_set';
newDatasetPath = './train_small_pulito';

data = readtable(csvPath);
imageNames = data{:, 1};
classes = data{:, 2};

[classCounts, classNames] = groupcounts(classes);

if ~exist(newDatasetPath, 'dir')
    mkdir(newDatasetPath);
end

uniqueClasses = unique(classNames);
for i = 1:length(uniqueClasses)
    folderPath = strcat(newDatasetPath, '/', num2str(i-1));

    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    className = uniqueClasses(i);
    classIndices = (classes == className);
    classImages = imageNames(classIndices);

    for j = 1:length(classImages)
        imagePath = fullfile(datasetPath, classImages{j});
        destFile = fullfile(folderPath, classImages{j});
        if exist(imagePath, 'file')
            copyfile(imagePath, destFile);
        else
            fprintf('File non trovato: %s\n', srcFile);
        end
    end
end

disp('Operazione completata.');