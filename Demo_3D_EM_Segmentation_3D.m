%% This code is written for Image Segmentation (EM algorithm)
%contact: kamruleeekuet@gmail.com
% Close and Clear all the figures and the workspace & Command windows 
clc;
clear all;
close all;
imtool close all;
tic;

%% Set the number of clusters and Stopping Threshold
Number_of_Cluster=3;
StoppingThreshold=0.001;

%% Set All the Image Paths

whichData=5;

currentDIrrectory=pwd;
Datapath=strcat(currentDIrrectory,'\P2_data\');
whichImage_T1='T1.nii';
whichImage_FLAIR='T2_FLAIR.nii';
whichImage_GT='LabelsForTesting.nii';

fullpath_T1=strcat(Datapath,num2str(whichData),'\',whichImage_T1);
fullpath_FLAIR=strcat(Datapath,num2str(whichData),'\',whichImage_FLAIR);
fullpath_GT=strcat(Datapath,num2str(whichData),'\',whichImage_GT);

%% Read MRI Images 
image_T1=niftiread(fullpath_T1);
image_FLAIR=niftiread(fullpath_FLAIR);
image_GT=niftiread(fullpath_GT);

%% Take only Brain Region (ROI) and Make Vector of all the slices for 3D Implementation.
% TotalElemects=0;
start=1;
Increment=0;
for whichSlice=1:1:length(image_GT(1,1,:))
    % Take one slice for each iteration to create Vector.
    slice_T1=image_T1(:,:,whichSlice);
    slice_FLAIR=image_FLAIR(:,:,whichSlice);
    slice_GT=image_GT(:,:,whichSlice);
    % Creating Barin Mask
    Brain_Mask=slice_GT;
    Brain_Mask(Brain_Mask==1)=255;
    Brain_Mask(Brain_Mask==2)=255;
    Brain_Mask(Brain_Mask==3)=255;
    %Taking the index from 2D slice that have ROI using GT ROI.
    temp_OnlyBrainRegionIndex=find(Brain_Mask==255);
    OnlyBrainRegionIndex{whichSlice}=temp_OnlyBrainRegionIndex; %Store index for the reconstruction of Image
    [elements,~]=size(temp_OnlyBrainRegionIndex); % How Many pixels coming from current slice
%     TotalElemects=TotalElemects+elements;
    T1_Image_Brain=slice_T1(temp_OnlyBrainRegionIndex); % Take only ROI of T1 Brain MRI and exclude background
    FLAIR_Image_Brain=slice_FLAIR(temp_OnlyBrainRegionIndex); % Take only ROI of FLAIR Brain MRI and exclude background
    bimodelImage_Slice_2D=[double(T1_Image_Brain),double(FLAIR_Image_Brain)];
    ended=elements+Increment;
    Data_Vector_3D(start:1:ended,:)=bimodelImage_Slice_2D;
    start=ended+1;
    Increment=ended;
end

%% Initialization for the EM based Segmentation using k-mean clustering
[cluster_indices,cluster_center]=kmeans(Data_Vector_3D,Number_of_Cluster,'MaxIter',1000,'Replicates',10);
% To fixed class label e.g. CSF=1, GM=2 and WM=3.
[new_cluster_center,Slice_ROI_Label] = sort(cluster_center(:,1));
new_cluster_indices=zeros(length(cluster_indices),1);
for Cluster_Label=1:1:Number_of_Cluster
    temp_Label=Slice_ROI_Label(Cluster_Label);
    index_temp_Var=find(cluster_indices==temp_Label);   
    new_cluster_indices(index_temp_Var)=Cluster_Label;
end

%% Initialization of the mean, shared proportions of GMM and Covariance.
for i=1:1:Number_of_Cluster
   Index_same_Cluster{i}=find(new_cluster_indices==i);
   Data_same_Cluster{i}=Data_Vector_3D(Index_same_Cluster{i},:);
   mean_GMM(i,:)=(1/length(Index_same_Cluster{i})).*(sum(Data_same_Cluster{i}));
   proportion_GMM(i)=(length(Index_same_Cluster{i}))/length(Data_Vector_3D(:,1));
   x=Data_same_Cluster{i};
   cov=zeros(2,2);
%    disp(covariance)
   for k=1:1:length(x)
       cov=cov+((x(k,:))'-(mean_GMM(i,:))')*((x(k,:))'-(mean_GMM(i,:))')';
   end
   cov=cov.*(1/length(Index_same_Cluster{i}));
   covariance{i}=cov;
   % Checking either cov is positive_definite or not!!!!!! If not, how to
   % solve? Link will provide by the code. Or see the below link (url).
   [~,positive_definite] = cholcov(cov);
   if positive_definite~=0
       disp(strcat('Sorry, Your calculated ', num2str(i),'th',' Coveriance Matrix is not positive definite!!'));
       pause(2)
       url = 'https://pdfs.semanticscholar.org/7d4a/2da54c78cf62a2e8ea60e18cef35ab0d5e25.pdf';
       web(url)
   end
%    disp(covariance)
%    disp(positive_definite)
%    pause(5)
end

%% Expectation Maximization Algorithm
Number_of_Iteration=1;
disp('---------------------Processing--------------------')

while(1)
    % Expectation which is evaluating the responsibilities using the current parameter values
    GM=Gaussian_Mixture(Data_Vector_3D,mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
    sum_all_Cluster = sum(GM,2)+eps;
    loglikelihood_Current=sum(log(sum_all_Cluster));
    latentVariable=GM./sum_all_Cluster; %Posterior Probability

    % % Maximization which is re-estimate the parameters using the current responsibilities
    for cluster=1:1:Number_of_Cluster
        mean_GMM(cluster,:)=(sum(latentVariable(:,cluster).*Data_Vector_3D))./(sum(latentVariable(:,cluster)));
        numerator = (repmat(latentVariable(:,cluster), 1, 2).* (Data_Vector_3D - mean_GMM(cluster,:)))' * ((Data_Vector_3D - mean_GMM(cluster,:)));
        denominator = sum(latentVariable(:,cluster));
        covariance{i}= numerator / denominator;
        % [~,positive_definite] = cholcov(cov);
        % if positive_definite~=0
        %    disp(strcat('Sorry, Your calculated ', num2str(i),'th',' Coveriance Matrix is not positive definite!!'));
        %    pause(2)
        %    url = 'https://pdfs.semanticscholar.org/7d4a/2da54c78cf62a2e8ea60e18cef35ab0d5e25.pdf';
        %    web(url)
        % end
        proportion_GMM(cluster)=sum(latentVariable(:,cluster))/length(Data_Vector_3D(:,1));
    end

    %% Stopping Criterion fixation
    GM=Gaussian_Mixture(Data_Vector_3D,mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
    sum_all_Cluster = sum(GM,2)+eps;
    loglikelihood_Updated=sum(log(sum_all_Cluster));

    difference_loglikelihood=loglikelihood_Updated-loglikelihood_Current;
    Store_Difference_loglikelihood(Number_of_Iteration)=difference_loglikelihood;

    disp(['Error--> ','Iteration = ',num2str(Number_of_Iteration),' --> ',num2str(difference_loglikelihood)]); % Display difference_loglikelihood for each Iteration

    if(abs(difference_loglikelihood)<StoppingThreshold) 
        break; 
    end
    
    Number_of_Iteration=Number_of_Iteration+1;
end
disp('-------------------Process DONE!!!--------------------')

%% Graphical Presentation of Error VS Iterations
figure()
plot(Store_Difference_loglikelihood);
xlabel('Iterations');
ylabel('Error (Difference between loglikelihood)');
xlim([0 50])
title('Error vs Iteration in 3D GMM EM')
grid on;

%% Display the Number of Iterations after converging to the given Stopping Threshold
disp(['Number of Iterations Require=',num2str(Number_of_Iteration)])

%% Pixels Classifications for the Segmenation
% PixelClassification=zeros(length(Data_Vector_3D(:,1)),1);
for data=1:1:length(Data_Vector_3D(:,1))
  MixturePDF=Gaussian_Mixture(Data_Vector_3D(data,:),mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
  Label=find(MixturePDF==max(MixturePDF));
  PixelClassification(data)=Label;
end

%% Split PixelClassification into individual Slice having same numbers of ROI pixels as OnlyBrainRegionIndex
PixelClassification=PixelClassification';
starting=1;
ending=0;
Number_of_Labeled_Slice=0;
for i=1:1:length(image_GT(1,1,:))
    Slice_ROI_Index=OnlyBrainRegionIndex{i}; 
    Length_Slice_ROI_Index=length(Slice_ROI_Index(:,1));
    if Length_Slice_ROI_Index~=0
        ending=ending+Length_Slice_ROI_Index;
        Slice_wise_ROI_Label{i}=PixelClassification(starting:ending);
        starting=starting+Length_Slice_ROI_Index;
        Number_of_Labeled_Slice=Number_of_Labeled_Slice+1;
    end   
end

%% Segmented Image recover
[rows,columns]=size(image_GT(:,:,1));
Segmented_Image_3D=zeros(rows,columns,length(image_GT(1,1,:)));

for i=1:1:Number_of_Labeled_Slice
    Slice_ROI_Index=OnlyBrainRegionIndex{i};
    Slice_ROI_Label=Slice_wise_ROI_Label{i};
    Current_Slice_1D=zeros(columns*rows,1);
    for k=1:1:length(Slice_ROI_Label(:,1))
        Current_Slice_1D(Slice_ROI_Index(k))=Slice_ROI_Label(k);
    end
    Segmented_Image_3D(:,:,i)=reshape(Current_Slice_1D,[rows,columns]);
end

%% Quantitative and Qualitative Visualization and display of the segmented Results using EM and GMM
figure()
for i=1:1:length(image_GT(1,1,:))
    CurrentSlice=Segmented_Image_3D(:,:,i);
    slice_GT=image_GT(:,:,i);
    
    img_CSF=(CurrentSlice==1);
    img_GM=(CurrentSlice==2);
    img_WM=(CurrentSlice==3);
    Segmented_image=cat(3,img_CSF,img_GM,img_WM); %CSF=Red, GM=Green and WM=Blue
    
    GT_CSF=(slice_GT==1);
    GT_GM=(slice_GT==2);
    GT_WM=(slice_GT==3);
    GT_image=cat(3,GT_CSF,GT_GM,GT_WM); %CSF=Red, GM=Green and WM=Blue
    
    %% DSC Calculation for each Tissue slice by slice
    DSC_CSF=Calculate_DiceCoefficient(double(slice_GT==1),img_CSF);
    DSC_CSF_Store(i)=DSC_CSF;
    DSC_GM=Calculate_DiceCoefficient(double(slice_GT==2),img_GM);
    DSC_GM_store(i)=DSC_GM;
    DSC_WM=Calculate_DiceCoefficient(double(slice_GT==3),img_WM);
    DSC_WM_store(i)=DSC_WM;
    disp(['-----------Dice Co-eeficient for Slice = ',num2str(i),'-----------'])
    disp(['DSC for CSF=',num2str(DSC_CSF)])
    disp(['DSC for GM=',num2str(DSC_GM)])
    disp(['DSC for WM=',num2str(DSC_WM)])
    
%% Qualitative Visualization
    imshowpair(Segmented_image,GT_image,'montage')
    title(['Segmented (Left) and GT (Right) for slice = ',num2str(i)])
    xlabel(['DSC for CSF=',num2str(DSC_CSF),', DSC for GM=',num2str(DSC_GM),', and DSC for WM=',num2str(DSC_WM)])
    pause(1)
    
    end

%% Display Avg. Qauntitative Performance (DSC)
disp(['Avg. DSC for CSF=',num2str(nanmean(DSC_CSF_Store)),', Avg. DSC for GM=',num2str(nanmean(DSC_GM_store)),', and avg. DSC for WM=',num2str(nanmean(DSC_WM_store))]);

toc;

%% Multi-variate Gaussian Mixture PDF Function
function GMM=Gaussian_Mixture(Data_Vector_3D,mean_GMM,covariance,proportion_GMM,HowManyCluster)
for i=1:1:HowManyCluster
    GMM(:,i)=proportion_GMM(i).*mvnpdf(Data_Vector_3D,mean_GMM(i,:),covariance{i});
end
end
%% --------------------- THE END-----------------------