function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1     %灰度图
    sz = size(imgs);    %获取图片尺寸
    sz = sz - mod(sz, modulo);  %图片尺寸对modulo取余，减去这个余数，使图片尺寸可以正好整除modulo
    imgs = imgs(1:sz(1), 1:sz(2)); %得到新尺寸图片
else        %彩色图
    tmpsz = size(imgs); %获取图片尺寸
    sz = tmpsz(1:2);     %图片的height，width
    sz = sz - mod(sz, modulo);  % height和width对modulo取余，减去这个余数，使图片尺寸可以正好整除modulo
    imgs = imgs(1:sz(1), 1:sz(2),:); %得到新的尺寸的图片
end

