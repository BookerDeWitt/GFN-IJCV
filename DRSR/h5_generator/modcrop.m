function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1     %�Ҷ�ͼ
    sz = size(imgs);    %��ȡͼƬ�ߴ�
    sz = sz - mod(sz, modulo);  %ͼƬ�ߴ��moduloȡ�࣬��ȥ���������ʹͼƬ�ߴ������������modulo
    imgs = imgs(1:sz(1), 1:sz(2)); %�õ��³ߴ�ͼƬ
else        %��ɫͼ
    tmpsz = size(imgs); %��ȡͼƬ�ߴ�
    sz = tmpsz(1:2);     %ͼƬ��height��width
    sz = sz - mod(sz, modulo);  % height��width��moduloȡ�࣬��ȥ���������ʹͼƬ�ߴ������������modulo
    imgs = imgs(1:sz(1), 1:sz(2),:); %�õ��µĳߴ��ͼƬ
end

