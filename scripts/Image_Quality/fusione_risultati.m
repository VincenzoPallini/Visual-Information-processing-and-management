piqe_file = 'risultati_PIQE.txt';
brisque_file = 'risultati_brisque.txt';

piqe_data = fileread(piqe_file);
brisque_data = fileread(brisque_file);

piqe_pattern = '(val_\d+\.jpg)\s+PIQE Score:\s+(\d+\.\d+)';
piqe_matches = regexp(piqe_data, piqe_pattern, 'tokens');
piqe_table = cell2table(reshape([piqe_matches{:}], 2, [])', 'VariableNames', {'File', 'PIQE_Score'});
piqe_table.PIQE_Score = str2double(piqe_table.PIQE_Score);

brisque_pattern = '(val_\d+\.jpg)\s+BRISQUE:\s+(\d+\.\d+)';
brisque_matches = regexp(brisque_data, brisque_pattern, 'tokens');
brisque_table = cell2table(reshape([brisque_matches{:}], 2, [])', 'VariableNames', {'File', 'BRISQUE_Score'});
brisque_table.BRISQUE_Score = str2double(brisque_table.BRISQUE_Score);

merged_table = outerjoin(piqe_table, brisque_table, 'Keys', 'File', 'MergeKeys', true);

merged_table = merged_table(~any(ismissing(merged_table), 2), :);

merged_table = sortrows(merged_table, 'File');

output_file = 'immagini_comuni.txt';
fileID = fopen(output_file, 'w');
for i = 1:height(merged_table)
    fprintf(fileID, '%s PIQE: %.2f BRISQUE: %.2f\n', merged_table.File{i}, merged_table.PIQE_Score(i), merged_table.BRISQUE_Score(i));
end
fclose(fileID);

num_images = height(merged_table);
fileID = fopen(output_file, 'a');
fprintf(fileID, '\nTotale immagini: %d\n', num_images);
fclose(fileID);

fprintf('File di output creato: %s\n', output_file);
