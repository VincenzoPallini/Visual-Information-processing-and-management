function blurScore = BlurDetect(img)
    if size(img,3) > 1
        gray_img = rgb2gray(uint8(img));
    else
        gray_img = uint8(img);
    end
    
    % Applica il filtro Laplaciano
    laplacian = fspecial('laplacian', 0);
    lap_img   = imfilter(double(gray_img), laplacian, 'replicate');
    
    blurScore = std2(lap_img)^2;
end
