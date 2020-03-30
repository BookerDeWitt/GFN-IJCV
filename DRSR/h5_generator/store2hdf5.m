function [curr_dat_sz, curr_lab_db_sz, curr_lab_sz] = store2hdf5(filename, data, labels_db, labels, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) beforehand
  % *label* is D*N matrix of labels (D labels per sample) 
  % *create* [0/1] specifies whether to create file newly or to append to previously created file, useful to store information in batches when a dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, 
  % if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  % if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  % chunksz (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations 

  % verify that format is right
  dat_dims=size(data);          %%data* is W*H*C*N
  lab_db_dims=size(labels_db);
  lab_dims=size(labels);
  num_samples=dat_dims(end); %dat_dims���һά����N

  assert(lab_dims(end)==num_samples, 'Number of samples should be matched between data and labels');%���ݺͱ�ǩ��Ӧ�����
  assert(lab_db_dims(end)==num_samples, 'Number of samples should be matched between data and labels');

  
  if ~exist('create','var')  %������ʱ�贴��
    create=true;
  end

  
  if create      %����ģʽ
    %fprintf('Creating dataset with %d samples\n', num_samples);
    if ~exist('chunksz', 'var')
      chunksz=1000;
    end
    if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
    end     
    
    %����ѹ�����ݼ���filenameΪHDF5�ļ�����/dataΪҪ����HDF5�ļ������ݼ����ƣ�
    %[dat_dims(1:end-1)Inf]���ݼ���С��'Datatype', 'single'�������͵�����
    %'ChunkSize', [dat_dims(1:end-1) chunksz])�ֿ��С
    h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number
    h5create(filename, '/label_db', [lab_db_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_db_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/label', [lab_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_dims(1:end-1) chunksz]); % width, height, channels, number 
    if ~exist('startloc','var')   %������ʱ��ʼ��Ϊȫ1����
      startloc.dat=[ones(1,length(dat_dims)-1), 1];  %����length(dat_dims)-1άȫ1����Ȼ����������1
      startloc.lab_db=[ones(1,length(lab_db_dims)-1), 1];
      startloc.lab=[ones(1,length(lab_dims)-1), 1];
    end 
    
  else  % append mode׷��ģʽ
    if ~exist('startloc','var')
      info=h5info(filename);   %�����й� filename ָ�������� HDF5 �ļ�����Ϣ
      prev_dat_sz=info.Datasets(1).Dataspace.Size;
      prev_lab_db_sz=info.Datasets(2).Dataspace.Size;
      prev_lab_sz=info.Datasets(3).Dataspace.Size;
      
      assert(prev_dat_sz(1:end-1)==dat_dims(1:end-1), 'Data dimensions must match existing dimensions in dataset'); %������ά�Ⱥ��Ѵ��ڵ�������ͬ��
      assert(prev_lab_db_sz(1:end-1)==lab_db_dims(1:end-1), 'Label_db dimensions must match existing dimensions in dataset');
      assert(prev_lab_sz(1:end-1)==lab_dims(1:end-1), 'Label dimensions must match existing dimensions in dataset');
      startloc.dat=[ones(1,length(dat_dims)-1), prev_dat_sz(end)+1];  %����֮ǰ�����ݺ��棬��ԭ���ݺ�һ����ʼ��ȫ1����
      startloc.lab_db=[ones(1,length(lab_db_dims)-1), prev_lab_db_sz(end)+1];
      startloc.lab=[ones(1,length(lab_dims)-1), prev_lab_sz(end)+1];
    end
  end

  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.dat, size(data));%������д��HDF5���ݼ�����startloc.datΪ��ʼд�������λ��
    h5write(filename, '/label_db', single(labels_db), startloc.lab_db, size(labels_db));  
    h5write(filename, '/label', single(labels), startloc.lab, size(labels));  
  end

  if nargout   %nargout���غ����������������
    info=h5info(filename);      %�����й� filename ָ�������� HDF5 �ļ�����Ϣ
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
    curr_lab_db_sz=info.Datasets(2).Dataspace.Size;
    curr_lab_sz=info.Datasets(3).Dataspace.Size;
  end
end
