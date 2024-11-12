clear all; close all;
network.classNames={{'NK','NK_dead','background'},{'NK','NK_dead','background'},{'DeadCell','background'}}; % 
network.labelIDs = {[255 128 0],[255,128,0],[255,0]};
networkFilePath=strcat('TrainedNet',filesep);

LoadTrainedNetFileName={'PlasticChip5xSpheroidNK.mat','PlasticChip5xNK.mat','PlasticChip5xDeadCell.mat'};
TestImagePath={strcat('DemoImages',filesep,'SpheroidNK'),strcat('DemoImages',filesep,'NK'),strcat('DemoImages',filesep,'DeadTumor')};

for i_net=1:numel(LoadTrainedNetFileName)
    LoadNet=load(strcat(networkFilePath,LoadTrainedNetFileName{i_net}));
    net_= assembleNetwork(layerGraph(LoadNet.MyNet));
    imageFiles = dir(fullfile(TestImagePath{i_net}, '*.tiff'));
    %
    [~,testName,~]=fileparts(TestImagePath{i_net});
    resultpath=strcat('Results',filesep,testName,filesep);
    mkdir(resultpath)
    %
    for i_sample = 1:numel(imageFiles)
        fileName = fullfile(TestImagePath{i_net}, imageFiles(i_sample).name);
        images_ = imread(fileName);
        images_RGB=Present_4chFrame2RGB(images_);

        [~,fname,~]=fileparts(fileName);
        pxdsResults_cate = semanticseg(images_, net_);
        for i_cate=1:length(network.classNames{i_net})-1
            segmResult_=double(pxdsResults_cate==network.classNames{i_net}{i_cate});
            segmResult_edge=255*double(bwmorph(segmResult_,'dilate',1)-segmResult_>0);
            images_RGB(:,:,i_cate)=images_RGB(:,:,i_cate)+uint8(segmResult_edge);
        end
        imwrite(uint8(images_RGB),strcat(resultpath,fname,'.tiff'));
    end 
end


function ColorimgRaw=Present_4chFrame2RGB(Image_4ch)
RGB=zeros(size(Image_4ch));
%% Norm all channels
for i=1:4
    RGB(:,:,i)=255*ImageNorm(Image_4ch(:,:,i));
end
%% make some colors more significant (by satuation)
bri_weight=0.6;
flo_weight1=0.5;
flo_weight2=0.5;
flo_weight3=0.5;
ColorimgRaw=uint8(zeros(size(Image_4ch,1),size(Image_4ch,2),3));

ColorimgRaw(:,:,1)=uint8(flo_weight1*RGB(:,:,1)+bri_weight*RGB(:,:,4));
ColorimgRaw(:,:,2)=uint8(flo_weight3*RGB(:,:,3)+bri_weight*RGB(:,:,4));
ColorimgRaw(:,:,3)=uint8(flo_weight2*RGB(:,:,2)+bri_weight*RGB(:,:,4));

% white balance
grayImage = rgb2gray(ColorimgRaw); % Convert to gray so we can get the mean luminance.
redChannel = ColorimgRaw(:, :, 1);
greenChannel = ColorimgRaw(:, :, 2);
blueChannel = ColorimgRaw(:, :, 3);
meanR = mean2(redChannel);
meanG = mean2(greenChannel);
meanB = mean2(blueChannel);
meanGray = mean2(grayImage);
% Make all channels have the same mean
redChannel = uint8(double(redChannel) * meanGray / meanR);
greenChannel = uint8(double(greenChannel) * meanGray / meanG);
blueChannel = uint8(double(blueChannel) * meanGray / meanB);
% Recombine separate color channels into a single, true color RGB image.
ColorimgRaw = cat(3, redChannel, greenChannel, blueChannel);
end