% Caricamento delle feature del dataset scelto da file o creazione di nuove se non esistente
function features = loadFeatures(setName)
    net = resnet101; % resnet101 
    layer = 'pool5';
    inputImageSize  = net.Layers(1).InputSize;

    switch setName
        case 'unlabeled'
            imdsUnlabeled = loadSet('unlabeled');
            
            augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);
            
            if exist('../saved_files/feature_unlabeled.mat', 'file')
                load('../saved_files/feature_unlabeled.mat', 'features');
                disp('Feature caricate da file pre-esistente: feature_unlabeled.mat');
            else
                disp('Estrazione Features da Unlabeled Set in corso...');
                features = activations(net,augimdsUnlabeled,layer,'OutputAs','rows');
                save('../saved_files/feature_unlabeled.mat', 'features');
                disp('Feature salvate su file: feature_unlabeled.mat');
            end

        case 'train'
            imdsUnlabeled = loadSet('train');
            augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);
            
            if exist('../saved_files/feature_train.mat', 'file')
                load('../saved_files/feature_train.mat', 'features');
                disp('Feature caricate da file pre-esistente: feature_unlabeled.mat');
            else
                disp('Estrazione Features da Train Set in corso...');
                features = activations(net,augimdsUnlabeled,layer,'OutputAs','rows');
                save('../saved_files/feature_train.mat', 'features');
                disp('Feature salvate su file: feature_train.mat');
            end
            
        case 'validation'
            imdsUnlabeled = loadSet('validation');
            augimdsUnlabeled = augmentedImageDatastore(inputImageSize(1:2), imdsUnlabeled);
            
            if exist('../saved_files/feature_validation.mat', 'file')
                load('../saved_files/feature_validation.mat', 'features');
                disp('Feature caricate da file pre-esistente: feature_validation.mat');
            else
                disp('Estrazione Features da Validation Set in corso...');
                features = activations(net,augimdsUnlabeled,layer,'OutputAs','rows');
                save('../saved_files/feature_validation.mat', 'features');
                disp('Feature salvate su file: feature_validation.mat');
            end
    end
end

