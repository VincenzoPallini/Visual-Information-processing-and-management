function analizza_set_BRISQUE()
    folder_path = '..\..\image_sets\val_set_degraded';

    files = dir(fullfile(folder_path, '*.jpg'));

    threshold = 40;

    output_file = "risultati_brisque.txt";

    brisqueScores = zeros(length(files), 1);

    imagesDegradate = [];

    fid = fopen(output_file, 'w');
    fprintf(fid, "Risultati Analisi BRISQUE:\n");
    fprintf(fid, "-----------------------------------------------\n");

    for i = 1:length(files)
        imgPath = fullfile(folder_path, files(i).name);
        img = imread(imgPath);

        score = measureBRISQUEScore(img);

        brisqueScores(i) = score;

        fprintf('Immagine: %s => BRISQUEScore: %.4f\n', files(i).name, score);

        fprintf(fid, 'Immagine: %s => BRISQUEScore: %.4f\n', files(i).name, score);

        if score > threshold
            imagesDegradate = [imagesDegradate; ...
                struct('name', files(i).name, 'score', score)];
        end
    end

    if ~isempty(imagesDegradate)
        imagesDegradate = struct2table(imagesDegradate);
        imagesDegradate = sortrows(imagesDegradate, 'score', 'descend');

        fprintf(fid, '\nElenco delle immagini considerate "Bassa qualità" :\n');
        fprintf(fid, "-----------------------------------------------\n");

        for i = 1:height(imagesDegradate)
            fprintf('%-40s BRISQUE: %.4f\n', ...
                imagesDegradate.name{i}, imagesDegradate.score(i));

            fprintf(fid, '%-40s BRISQUE: %.4f\n', ...
                imagesDegradate.name{i}, imagesDegradate.score(i));
        end

        fprintf(fid, "-----------------------------------------------\n");
    else
        fprintf('\nNessuna immagine supera la soglia di BRISQUE = %.2f.\n', threshold);
        fprintf(fid, '\nNessuna immagine supera la soglia di BRISQUE = %.2f.\n', threshold);
    end

    fclose(fid);

    fprintf('\nRisultati salvati nel file: %s\n', output_file);
end
