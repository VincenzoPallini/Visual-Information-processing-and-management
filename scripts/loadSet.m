% Caricamento del dataset scelto da file o creazione di nuovo se non esistente
function imds = loadSet(setName)
    trainDir        = '../image_sets/train_set';
    valDir          = '../image_sets/val_set';
    valDegDir          = '../image_sets/val_set_degraded';

    switch setName
        case 'train'
            csvTrainPath = '../csv/train_set.csv';
            if exist('../saved_files/trainData.mat', 'file')
                load('../saved_files/trainData.mat', 'imds');
                disp('Dataset di training caricato da file pre-esistente: saved_files/trainData.mat');
            else
                disp('Creazione file trainData.mat in corso...');
                dataTrain = readtable(csvTrainPath);
                imageNames = dataTrain{:, 1};
                classes = dataTrain{:, 2};
            
                for j = 1:numel(imageNames)
                    imageNames{j} = fullfile(trainDir, imageNames{j});
                end
            
                imds = imageDatastore(imageNames, 'labels', categorical(classes));
            
                save('../saved_files/trainData.mat', 'imds');
                disp('Dataset di training salvato in saved_files/trainData.mat');
            end

        case 'unlabeled'
            csvUnlabeledPath = '../csv/train_unlabeled.csv';
            if exist('../saved_files/unlabeledData.mat', 'file')
                load('../saved_files/unlabeledData.mat', 'imds');
                disp('Dataset unlabeled caricato da file pre-esistente: saved_files/unlabeledData.mat');
            else
                disp('Creazione file unlabeledData.mat in corso...');
                dataTrain = readtable(csvUnlabeledPath);
                imageNames = dataTrain{:, 1};
                classes = dataTrain{:, 2};
            
                for j = 1:numel(imageNames)
                    imageNames{j} = fullfile(trainDir, imageNames{j});
                end
            
                imds = imageDatastore(imageNames, 'labels', categorical(classes));
            
                save('../saved_files/unlabeledData.mat', 'imds');
                disp('Dataset di training salvato in ../saved_files/unlabeledData.mat');
            end

        case 'validation'
            csvValPath = '../csv/val_info.csv';
            if exist('../saved_files/valData.mat', 'file')
                load('../saved_files/valData.mat', 'imds');
                disp('Dataset di validazione caricato da file pre-esistente: saved_files/valData.mat');
            else
                disp('Creazione file valData.mat in corso...');
                dataTrain = readtable(csvValPath);
                imageNames = dataTrain{:, 1};
                classes = dataTrain{:, 2};
            
                for j = 1:numel(imageNames)
                    imageNames{j} = fullfile(valDir, imageNames{j});
                end
            
                imds = imageDatastore(imageNames, 'labels', categorical(classes));
            
                save('../saved_files/valData.mat', 'imds');
                disp('Dataset di training salvato in saved_files/valData.mat');
            end

        case 'degraded'
            csvDegValPath = '../csv/val_info.csv';
            if exist('../saved_files/valDegData.mat', 'file')
                load('../saved_files/valDegData.mat', 'imds');
                disp('Dataset di validazione caricato da file pre-esistente: saved_files/valData.mat');
            else
                disp('Creazione file valDegData.mat in corso...');
                dataTrain = readtable(csvDegValPath);
                imageNames = dataTrain{:, 1};
                classes = dataTrain{:, 2};
            
                for j = 1:numel(imageNames)
                    imageNames{j} = fullfile(valDegDir, imageNames{j});
                end

                imds = imageDatastore(imageNames, 'labels', categorical(classes));
            
                save('../saved_files/valDegData.mat', 'imds');
                disp('Dataset di training salvato in saved_files/valData.mat');
            end
    end
end

