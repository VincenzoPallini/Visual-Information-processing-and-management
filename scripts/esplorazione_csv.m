% Parametri
csvPath = './../csv/train_set.csv';
datasetPath = './../image_sets/train_set';

data = readtable(csvPath);

imageNames = data{:, 1};
classes = data{:, 2};

disp(numel(imageNames))

[classCounts, classNames] = groupcounts(classes);

%% Distribuzione delle classi tramite grafico
figure (1);
bar(classNames, classCounts);
xlabel('Label');
xticks(0:5:length(classNames))
ylabel('Numero di immagini');
title('Distribuzione delle immagini per label');

%% Esempio di immagini per classe
figure (2);
uniqueClasses = unique(classNames);
for i = 1:length(uniqueClasses)
    className = uniqueClasses(i);
    classIndices = (classes == className);
    classImages = imageNames(classIndices);
   % classScores = scores(classIndices);
    for j = 1:min(50, length(classImages))
        imagePath = fullfile(datasetPath, classImages{j});
        img = imread(imagePath);
        subplot(5, 10, j);
        imshow(img);
        title(['Classe: ', num2str(className)]);
    end
    pause
end