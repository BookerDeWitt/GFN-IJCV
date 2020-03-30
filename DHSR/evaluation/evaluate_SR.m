function evaluate_SR(folder)
% -------------------------------------------------------------------------
%   Description:
%       Compute PSNR and IFC for SR
%       We convert RGB image to grayscale and crop boundaries for 'scale'
%       pixels
%
%   Input:
%       - img_GT        : Ground truth image
%       - img_HR        : predicted HR image
%       - scale         : upsampling scale
%       - compute_ifc   : evaluate IFC [default = 0 since it's slow]
%
%   Citation: 
%       Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks
%       Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, and Ming-Hsuan Yang
%       arXiv, 2017
%
%   Contact:
%       Wei-Sheng Lai
%       wlai24@ucmerced.edu
%       University of California, Merced
% -------------------------------------------------------------------------
    scale = 1
    HR_image_path = fullfile(folder, 'HR/')
    Blur_image_path = fullfile(folder ,'Results/')
    filepaths_HR = dir(fullfile(HR_image_path, '*.png')); 
    filepaths_BLur = dir(fullfile(Blur_image_path, '*.png')); 
    avg_psnr = 0
    for i = 1 : length(filepaths_HR)
           img_GT = imread(fullfile(HR_image_path,filepaths_HR(i).name));
           img_HR = imread(fullfile(Blur_image_path,filepaths_BLur(i).name));
    %% quantize pixel values
           img_GT = im2double(im2uint8(img_GT)); 
           img_HR = im2double(im2uint8(img_HR)); 
           img_GT = shave_bd(img_GT, scale);
           img_HR = shave_bd(img_HR, scale);
           PSNR = psnr(img_GT, img_HR);
           %sprintf("psnr: %d", PSNR)
           avg_psnr = PSNR + avg_psnr;
           %SSIM = ssim(img_GT, img_HR);
           %SSIM = 0;
    end
    avg_psnr = avg_psnr / length(filepaths_HR)
    sprintf("res : %d", avg_psnr)