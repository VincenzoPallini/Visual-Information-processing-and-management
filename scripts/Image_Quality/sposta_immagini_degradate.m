close all;

sourceFolder = '..\..\image_sets\val_set_degraded';

destinationFolder = fullfile(sourceFolder, "bassa_qualità");
if ~exist(destinationFolder, 'dir')
    mkdir(destinationFolder);
    fprintf('Creata cartella di destinazione: %s\n', destinationFolder);
end

textFiles = {
    '..\risultati_comuni.txt', ...
    '..\Immagini con blur.txt', ...
    '..\Immagini con rumore gaussiano.txt'
};


pattern = '(val_\d{6}\.jpg)';

allFilesToMove = {};

for i = 1:numel(textFiles)
    filePath = textFiles{i};
    
    fid = fopen(filePath, 'r');
    if fid == -1
        warning('Impossibile aprire il file: %s', filePath);
        continue;  
    end
    
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            tokens = regexp(line, pattern, 'tokens');
            if ~isempty(tokens)
                for k = 1:numel(tokens)
                    allFilesToMove{end+1} = tokens{k}{1}; 
                    
                end
            end
        end
    end
    
    fclose(fid);
end

allFilesToMove = unique(allFilesToMove);

countMoved = 0;
for i = 1:numel(allFilesToMove)
    fileName = allFilesToMove{i};  
    sourceFile = fullfile(sourceFolder, fileName);
    
    if exist(sourceFile, 'file')
        destinationFile = fullfile(destinationFolder, fileName);
        
        movefile(sourceFile, destinationFile);
        countMoved = countMoved + 1;
    else
        warning('File non trovato (non spostato): %s', sourceFile);
    end
end

fprintf('\nTotale immagini spostate in bassa_qualità: %d\n', countMoved);
fprintf('Cartella di destinazione: %s\n', destinationFolder);
fprintf('File di testo analizzati:\n');
for i = 1:numel(textFiles)
    fprintf('  - %s\n', textFiles{i});
end
