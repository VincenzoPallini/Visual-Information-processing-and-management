clear all

im = imread('..\..\image_sets\val_set_degraded\val_000016.jpg');
bri_orig = brisque(im);


BRI=[];
for sigma=0.01:0.01:3
    corrected=imgaussfilt(im, sigma);
    BRI=[BRI; sigma brisque(corrected)];
end
figure(), plot(BRI(:,1), BRI(:,2))

%% visualizzo miglior risultato
[~ , idx] = min(BRI(:,2));
corrected = imgaussfilt(im, BRI(idx,1));

bri_corrected = brisque(corrected);
figure()
subplot(1,2,1), imshow(im), title('Degradata')
subplot(1,2,2), imshow(corrected), title('Restaurata')



