clear all
clc
close all
rng(42)
addpath(genpath(pwd))

im1 = im2double(imread('..\..\image_sets\val_set\val_000033.jpg'));

imDegraded = im2double(imread('..\..\image_sets\val_set_degraded\val_000033.jpg'));

im2 = imnoise(im1,'gaussian',0.0,0.06);

figure(1),
subplot(1,3,1), imshow(im1), title('Originale');
subplot(1,3,2), imshow(im2), title('Rumore gaussiano (var=0.06)');
subplot(1,3,3), imshow(imDegraded), title('Immagine degradata');

fprintf('PSNR(im1 vs im2): %f\n', psnr(im1, im2));
fprintf('PSNR(im1 vs imDegraded): %f\n', psnr(im1, imDegraded));
fprintf('PSNR(im2 vs imDegraded): %f\n', psnr(im2, imDegraded));
