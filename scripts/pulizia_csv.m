%% File Mancanti in Train Small
% Usato per capire quali immagini sono state rimosse dalla pulizia delle immagini di non cibo 
% che erano anche nel train_small
data = readtable('../csv/new_train_small.csv');
datasetPath = '../train_set';

imageNames = data{:, 1};
classes = data{:, 2};

uniqueClasses = unique(classNames);
for i = 1:length(uniqueClasses)
    className = uniqueClasses(i);
    classIndices = (classes == className);
    classImages = imageNames(classIndices);

    for j = 1:length(classImages)
        try
        imagePath = fullfile(datasetPath, classImages{j});
        img = imread(imagePath);
        catch
            disp(imagePath)
        end
    end
end

%% Rimozione File Mancanti da Train Small
data = readtable('../csv/train_small.csv');

imageNames = data{:, 1};

fileID = fopen('../csv/files_to_remove.txt', 'r');
filesToRemove = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);
filesToRemove = filesToRemove{1};
filteredData = data(~ismember(imageNames, filesToRemove), :);

writetable(filteredData, '../csv/train_small_pulito.csv');

%% File Mancanti in Train Unlabeled
% Usato per capire quali immagini sono state rimosse dalla pulizia delle immagini di non cibo 
data = readtable('../csv/train_unlabeled.csv');
datasetPath = '../train_set';

fid = fopen('files_to_remove_from_unlabeled.txt', 'a+');

imageNames = data{:, 1};

for j = 1:length(imageNames)
    try
    imagePath = fullfile(datasetPath, imageNames{j});
    img = imread(imagePath);
    catch
        fprintf(fid, '%s\n', imagePath);
    end
end
fclose(fid);


%% Rimozione File Mancanti da Train Unlabeled
data = readtable('../csv/train_unlabeled.csv');

imageNames = data{:, 1};

fileID = fopen('../csv/files_to_remove_from_unlabeled.txt', 'r');
filesToRemove = textscan(fileID, '%s', 'Delimiter', '\n');
fclose(fileID);
filesToRemove = filesToRemove{1};
filteredData = data(~ismember(imageNames, filesToRemove), :);

writetable(filteredData, '../csv/train_unlabeled_pulito.csv');