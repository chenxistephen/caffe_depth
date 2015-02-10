function [] = createDataset_depthNet_TrainTest()
recreateDataset = 1;
rewriteTrainset = 0;
rewriteTestset = 1;
rewriteInputData = 1;
rewriteLabel = 1;
%%
dataPath = '/scratch/stephenchen/Dropbox/ShapeAttribute/depthNet/';
modelPath = '/scratch/stephenchen/Dropbox/ShapeAttribute/depthNet/DepthScan/';
hdfPath = [dataPath 'hdf5/']; if ~isdir(hdfPath) mkdir(hdfPath); end;
modelList = dir([modelPath '*.mat']);
modelNum = length(modelList);
trainIdx = [1:modelNum];
maxDepth = 1000;
inputH = 500;
inputW = 500;
labelH = 123;
labelW = 123;
trainTestSplit = 6/4;
trainRatio = trainTestSplit/(1+trainTestSplit);
%% Create HDF5 dataset
allDatasetName = sprintf('%sAll_DepthNet_HDF_%dx%d.h5',hdfPath, inputH, inputW);
allInfo = h5info(allDatasetName);
dataSize = allInfo.Datasets(1).Dataspace.Size;
inputH = dataSize(1); inputW = dataSize(2);
allDataNum = dataSize(4);
allTrainNum = ceil(allDataNum * trainRatio);
allTestNum = allDataNum - allTrainNum;
%%
trainSubsetNum = 30;
testSubsetNum = 20;
trainHFolder = sprintf('%sTrain_Batch_%dx%d/',hdfPath, inputH, inputW); if ~isdir(trainHFolder) mkdir(trainHFolder); end;
testHFolder = sprintf('%sTest_Batch_%dx%d/',hdfPath, inputH, inputW); if ~isdir(testHFolder) mkdir(testHFolder); end;
%%
subsetTrainNum = ceil(allTrainNum / trainSubsetNum);
if rewriteTrainset
    for dd = 1: trainSubsetNum
        trainDatasetName = sprintf('%sTrainHDF_DepthNet_%d_%dx%d.h5',trainHFolder, dd, inputH, inputW);
        fprintf('Creating %s',trainDatasetName);
        if ~exist(trainDatasetName,'file') || recreateDataset
            delete(trainDatasetName);            
        end;
        if rewriteInputData
            h5create(trainDatasetName,'/data',[inputH inputW 3 subsetTrainNum],'Datatype','single','ChunkSize',[inputH inputW 1 1]);            
        end;
        if rewriteLabel
            h5create(trainDatasetName,'/label',[labelH labelW 1 subsetTrainNum],'Datatype','single','ChunkSize',[labelH labelW  1 1]); % depth
        end;
        for idxInSubset = 1 : subsetTrainNum
            idxInAll = mod((dd-1) * subsetTrainNum + idxInSubset-1, allTrainNum)+1;
            if rewriteInputData
                trainData = h5read(allDatasetName, '/data', [1 1 1 idxInAll],[inputH inputW 3 1]);
                h5write(trainDatasetName, '/data', trainData, [1 1 1 idxInSubset],[inputH inputW 3 1]);
            end;
            %clear trainData
            if rewriteLabel
                trainLabel = h5read(allDatasetName, '/label', [1 1 1 idxInAll],[inputH inputW 1 1]);
                trainLabel = imresize(trainLabel,[labelH labelW]);
                trainLabel(find(trainLabel>5)) = 0;
                trainLabel(find(trainLabel<0)) = 0;                
                h5write(trainDatasetName, '/label', trainLabel, [1 1 1 idxInSubset],[labelH labelW  1 1]);
            end;
            if mod(idxInSubset,100) == 0
                disp(idxInSubset);
            end;
        end;
    end;
end;
%%
subsetTestNum = ceil(allTestNum / testSubsetNum);
if rewriteTestset
    for dd = 1 : testSubsetNum
        testDatasetName = sprintf('%sTestHDF_DepthNet_%d_%dx%d.h5',testHFolder, dd, inputH, inputW);
        fprintf('Creating %s',testDatasetName);
        if ~exist(testDatasetName,'file') || recreateDataset
            delete(testDatasetName);

        end;
        if rewriteInputData
            h5create(testDatasetName,'/data',[inputH inputW 3 subsetTestNum],'Datatype','single','ChunkSize',[inputH inputW 1 1]);
        end;
        if rewriteLabel
            h5create(testDatasetName,'/label',[labelH labelW 1 subsetTestNum],'Datatype','single','ChunkSize',[labelH labelW 1 1]); % depth
        end;
        for idxInSubset = 1 : subsetTestNum
            idxInAll = allTrainNum + mod((dd-1) * subsetTestNum + idxInSubset -1 , allTestNum) +1;
            if rewriteInputData
                testData = h5read(allDatasetName, '/data', [1 1 1 idxInAll],[inputH inputW 3 1]);
                h5write(testDatasetName,'/data',  testData, [1 1 1 idxInSubset],[inputH inputW 3 1]);
            end;
            %clear testData
            if rewriteLabel
                testLabel = h5read(allDatasetName, '/label', [1 1 1 idxInAll],[inputH inputW 1 1]);
                testLabel = imresize(testLabel,[labelH labelW]);                                
                testLabel(find(testLabel>5)) = 0;
                testLabel(find(testLabel<0)) = 0; 
                h5write(testDatasetName, '/label', testLabel, [1 1 1 idxInSubset],[labelH labelW  1 1]);
            end;
            %clear testLabel
            if mod(idxInSubset,100) == 0
                disp(idxInSubset); end;
        end;
    end;
end;
disp('');