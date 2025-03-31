function analizza_set_PIQE()
    folder_path = '..\image_sets\val_set_degraded';
    files = dir(fullfile(folder_path, '*.jpg'));
    
    results = struct('filename', {}, 'piqe_score', {}, 'is_degraded', {});
    
    threshold = 65;  
    
    for i = 1:length(files)
        img_path = fullfile(folder_path, files(i).name);
        img = imread(img_path);
        
        score = piqe(img);
        
        results(i).filename = files(i).name;
        results(i).piqe_score = score;
        results(i).is_degraded = score > threshold;
        
        fprintf('Analisi %d/%d: %s => PIQE Score: %.2f\n', ...
            i, length(files), files(i).name, score);
    end
    
    resultsTable = struct2table(results);
    
    resultsTable = sortrows(resultsTable, 'piqe_score', 'descend');
    
    degraded = resultsTable(resultsTable.is_degraded, :);
    
    fprintf('\nImmagini degradate (Score > %.2f):\n', threshold);
    fprintf('--------------------------------\n');
    
    if height(degraded) > 0
        for i = 1:height(degraded)
            fprintf('%s (PIQE Score: %.2f)\n', ...
                degraded.filename{i}, degraded.piqe_score(i));
        end
    else
        fprintf('Nessuna immagine degradata trovata.\n');
    end
    
    fprintf('\nStatistiche:\n');
    fprintf('Media PIQE: %.2f\n', mean(resultsTable.piqe_score));
    fprintf('Mediana PIQE: %.2f\n', median(resultsTable.piqe_score));
    fprintf('Immagini degradate: %d/%d\n', ...
        height(degraded), height(resultsTable));
    
    if height(degraded) > 0
        fileID = fopen('risultati_PIQE.txt', 'w');
        
        fprintf(fileID, 'Risultati Analisi PIQE - Immagini Degradate\n');
        fprintf(fileID, '==========================================\n\n');
        fprintf(fileID, 'Soglia utilizzata: %.2f\n', threshold);
        fprintf(fileID, 'Numero totale immagini analizzate: %d\n', height(resultsTable));
        fprintf(fileID, 'Numero immagini degradate trovate: %d\n\n', height(degraded));
        fprintf(fileID, 'Dettaglio immagini degradate:\n');
        fprintf(fileID, '-------------------------\n\n');
        
        for i = 1:height(degraded)
            fprintf(fileID, '%d. File: %s\n', i, degraded.filename{i});
            fprintf(fileID, '   PIQE Score: %.2f\n\n', degraded.piqe_score(i));
        end
        
        fprintf(fileID, '\nStatistiche Generali:\n');
        fprintf(fileID, '-------------------\n');
        fprintf(fileID, 'Media PIQE: %.2f\n', mean(resultsTable.piqe_score));
        fprintf(fileID, 'Mediana PIQE: %.2f\n', median(resultsTable.piqe_score));
        
        fclose(fileID);
        
        writetable(degraded, 'risultati_PIQE.csv');
        
        fprintf('\nRisultati salvati in "risultati_PIQE.txt" e "risultati_PIQE.csv"\n');
    else
        fprintf('\nNessun risultato da salvare poiché non sono state trovate immagini degradate.\n');
    end
end