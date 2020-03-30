function rain_hdf5_generator(folder)
save_root =fullfile(folder, 'Rain_HDF5');

if ~isdir(save_root)
   mkdir(save_root)
end

scale = 4;

size_label = 512;
size_input = size_label/scale;
stride = 96;

%% generate data
train_sets = dir(fullfile(folder,'without_rain','*.jpg'));
test_sets = dir(fullfile(folder,'with_rain','*.jpg'));
lens = length(train_sets);
% downsizes= [0.65,0.8,1];
%downsizes = [1];
for n=1:40 %n
data = zeros(size_input, size_input, 3, 1);
label_db = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
count = 0;
margain = 0;    
for index = (n-1)*600+1 :n*600
index2 = ceil(index /6);
index, index2
HR_image_path = fullfile(folder, 'without_rain', train_sets(index2).name);
Blur_image_path = fullfile(folder, 'with_rain', test_sets(index).name);

%     for downsize = 1:length(downsizes)
                image = imread(HR_image_path);
                image_Blur = imread(Blur_image_path);
%                 image = imresize(image,downsizes(downsize),'bicubic');
%                 image_Blur = imresize(image_Blur,downsizes(downsize),'bicubic');
                if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);
                    image_Blur = im2double(image_Blur);
                    HR_label = modcrop(image, scale);
                    Blur_label = modcrop(image_Blur, scale);
                    [hei,wid, c] = size(HR_label);

                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            %Crop HR patch
                            HR_patch_label = HR_label(x : x+size_label-1, y : y+size_label-1, :);
                            [dx,dy] = gradient(HR_patch_label);
                            gradSum = sqrt(dx.^2 + dy.^2);
                            gradValue = mean(gradSum(:));
                            if gradValue < 0
                                continue;
                            end    
                            %Crop Blur patch
                            Blur_patch_label = Blur_label;
                            
                            LR_BLur_input = Blur_patch_label;
                            Deblur_label = imresize(HR_patch_label,1/scale,'bicubic');
%                             if mod(index ,2000) == 0
%                                 figure;
%                                 imshow(LR_BLur_input);
%                                 figure;
%                                 imshow(Deblur_label);
%                                 figure;
%                                 imshow(HR_patch_label);
%                             end
%                             if mod(index ,3000) == 0
%                                 close all;
%                             end
                            count=count+1;
                            data(:, :, :, count) = LR_BLur_input;
                            label_db(:, :, :, count) = Deblur_label;
                            label(:, :, :, count) = HR_patch_label;
                        end % end of y 
                    end % end of x
                end % end of if
%     end %end of downsize
end % end of index

order = randperm(count);
data = data(:, :, :, order);
label_db = label_db(:, :, :, order); 
label = label(:, :, :, order); 

%% writing to HDF5
savepath = fullfile(save_root ,sprintf('LR-GOPRO_x4_Part%d.h5', n));
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs_db = label_db(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_db', [1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_db, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
end % index fo n