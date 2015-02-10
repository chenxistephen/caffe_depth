function [] = comptue_mean_depthNet()
recreateDataset = 1;
dataPath = '/scratch/stephenchen/Dropbox/ShapeAttribute/depthNet/';
modelPath = '/scratch/stephenchen/Dropbox/ShapeAttribute/depthNet/DepthScan/';
hdfPath = [dataPath 'hdf5/']; if ~isdir(hdfPath) mkdir(hdfPath); end;
modelList = dir([modelPath '*.mat']);
modelNum = length(modelList);
startModelIdx = 1;
trainIdx = [startModelIdx:modelNum];
maxDepth = 1000;
dataH = 304; 
dataW = 228;
labelH = 74;
labelW = 55;
viewNumPerShape = 360;
dataNumPerSubset = 5760; % 500x500 * 1167==>3.3GB, 250x250*6000==>5GB
shapeNumPerSubset = dataNumPerSubset / viewNumPerShape;
%% Create HDF5 dataset
trainDatasetName = sprintf('%sAll_DepthNet_HDF_%dx%d.h5',hdfPath, dataH, dataW);
if ~exist(trainDatasetName,'file') || recreateDataset
    delete(trainDatasetName);
    h5create(trainDatasetName,'/data',[dataH dataW 3 Inf],'Datatype','single','ChunkSize',[dataH dataW 1 1]);
    h5create(trainDatasetName,'/label',[labelH labelW 1 Inf],'Datatype','single','ChunkSize',[labelH labelW  1 1]); % depth
end;
%%
subsetsFolder = sprintf('%sTrainTest_Batch_%dx%d/',hdfPath, dataH, dataW); if ~isdir(subsetsFolder) mkdir(subsetsFolder); end;
% viewNum = 360;
idxInH5 = 0;% (startModelIdx-1) * viewNum; %0;
subsetId = 0;
% %% temp:
% subsetName = sprintf('%sHDF_DepthNet_%d_%dx%d.h5',subsetsFolder, subsetId, dataH, dataW);
% delete(subsetName);
% h5create(subsetName,'/data',[dataH dataW 3 dataNumPerSubset],'Datatype','single','ChunkSize',[dataH dataW 1 1]);
% h5create(subsetName,'/label',[labelH labelW 1 dataNumPerSubset],'Datatype','single','ChunkSize',[labelH labelW  1 1]); % depth
%%
idxInSubset = dataNumPerSubset;%mod(idxInH5-1, dataNumPerSubset); % dataNumPerSubset; % initialize so that idx++ > dataNumPerSubset
writeSubset = 1;
for mm = trainIdx
    tic;
    fprintf('Model %d Started!\n',mm);
    modelName = modelList(mm).name;
    load([modelPath modelName],'Depth');
    viewNum = length(Depth.img);
    for ii = 1 : viewNum
        rgbd = Depth.img{ii};     
        [viewImage] = gen_rgb_image(rgbd);
        [s1 s2 dim] = size(viewImage);
        depthImage = zeros(s1, s2);
        depthImage(rgbd.pointPixelIds) = rgbd.depth;
        %%% imresize
        viewImage = imresize(viewImage,[dataH dataW]);
        depthImage = imresize(depthImage,[labelH labelW]);
        depthImage(find(depthImage>5)) = 0;
        depthImage(find(depthImage<0)) = 0;
        %%% Write to HDF
        idxInH5 = idxInH5 + 1;
        idxInSubset = idxInSubset +1;
        h5write(trainDatasetName,'/data',single(viewImage), [1 1 1 idxInH5], [size(viewImage) 1]);
        h5write(trainDatasetName,'/label',single(depthImage),[1 1 1 idxInH5], [size(depthImage) 1 1]);
        %%% Write to subset
        if writeSubset
            if idxInSubset > dataNumPerSubset
                idxInSubset = 1;
                subsetId = subsetId + 1;
                subsetName = sprintf('%sHDF_DepthNet_%d_%dx%d.h5',subsetsFolder, subsetId, dataH, dataW);
                fprintf('Creating %s','HDF_DepthNet_%d_%dx%d.h5', subsetId, dataH, dataW);
                %%%%
                delete(subsetName);
                h5create(subsetName,'/data',[dataH dataW 3 dataNumPerSubset],'Datatype','single','ChunkSize',[dataH dataW 1 1]);
                h5create(subsetName,'/label',[labelH labelW 1 dataNumPerSubset],'Datatype','single','ChunkSize',[labelH labelW  1 1]); % depth
            end;
            h5write(subsetName,'/data',single(viewImage), [1 1 1 idxInSubset], [size(viewImage) 1]);
            h5write(subsetName,'/label',single(depthImage),[1 1 1 idxInSubset], [size(depthImage) 1 1]);
            if mod(idxInSubset,100) == 0
                disp(idxInSubset); 
            end;
        end;
        fprintf('--View %d done\n', ii);
    end;
    
    fprintf('Model %d Done\n',mm);
    toc;
end;
disp('');