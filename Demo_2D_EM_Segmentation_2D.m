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

whichData=5; % Among 5 Images, you can select any one. 

currentDIrrectory=pwd;
Datapath=strcat(currentDIrrectory,'\P2_data\');
whichImage_T1='T1.nii';
whichImage_FLAIR='T2_FLAIR.nii';
whichImage_GT='LabelsForTesting.nii';

fullpath_T1=strcat(Datapath,num2str(whichData),'\',whichImage_T1);
fullpath_FLAIR=strcat(Datapath,num2str(whichData),'\',whichImage_FLAIR);
fullpath_GT=strcat(Datapath,num2str(whichData),'\',whichImage_GT);

%% Read MRI Images (T1, FLAIR and GT)
image_T1=niftiread(fullpath_T1);

image_FLAIR=niftiread(fullpath_FLAIR);

image_GT=niftiread(fullpath_GT);

%% Iterate slice by Slice
for whichSlice=1:1:length(image_T1(1,1,:))
    slice_T1=image_T1(:,:,whichSlice);
    slice_FLAIR=image_FLAIR(:,:,whichSlice);
    slice_GT=image_GT(:,:,whichSlice);
    
    if max(max(slice_GT))~=0 % Checking is there any foreground or not?
    %% Create BRAIN Mask to take only Brain Region
        Brain_Mask=slice_GT;
        Brain_Mask(Brain_Mask==1)=255;
        Brain_Mask(Brain_Mask==2)=255;
        Brain_Mask(Brain_Mask==3)=255;
        % figure()
        % imshow(Brain_Mask,[])

        %% Take only Brain Region
        OnlyBrainRegionIndex=find(Brain_Mask==255);

        T1_Image_Brain=slice_T1(OnlyBrainRegionIndex);

        FLAIR_Image_Brain=slice_FLAIR(OnlyBrainRegionIndex);

        bimodelImage_2D=[double(T1_Image_Brain),double(FLAIR_Image_Brain)];

        %% Initialization for the EM based Segmentation using k-mean clustering
        [cluster_indices,cluster_center]=kmeans(bimodelImage_2D,Number_of_Cluster,'MaxIter',1000,'Replicates',10);
        % To fixed class label e.g. CSF=1, GM=2 and WM=3.
        [new_cluster_center,I] = sort(cluster_center(:,1));
        new_cluster_indices=zeros(length(cluster_indices),1);
        for Cluster_Label=1:1:Number_of_Cluster
            temp_Label=I(Cluster_Label);
            index_temp_Var=find(cluster_indices==temp_Label);   
            new_cluster_indices(index_temp_Var)=Cluster_Label;
        end

        %% Initialization of the mean, shared proportions of GMM and Covariance.
        for i=1:1:Number_of_Cluster
           Index_same_Cluster{i}=find(new_cluster_indices==i);
           Data_same_Cluster{i}=bimodelImage_2D(Index_same_Cluster{i},:);
           mean_GMM(i,:)=(1/length(Index_same_Cluster{i})).*(sum(Data_same_Cluster{i}));
           proportion_GMM(i)=(length(Index_same_Cluster{i}))/length(bimodelImage_2D(:,1));
           x=Data_same_Cluster{i};
           cov=zeros(2,2);
           % disp(covariance)
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

        %% Store all the parameters before update using EM algorith
        mean_GMM_copy=mean_GMM;
        covariance_copy=covariance;
        proportion_GMM_copy=proportion_GMM;

        %% Expectation Maximizations Algorithm
        Number_of_Iteration=1;
        disp(['-----------Processing for Slice Number= ',num2str(whichSlice),'-----------'])
        while(1)
        % Expectation which is evaluating the responsibilities using the
        % current parameter values.
        GM=Gaussian_Mixture(bimodelImage_2D,mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
        sum_all_Cluster = sum(GM,2)+eps;
        loglikelihood_Current=sum(log(sum_all_Cluster));
        latentVariable=GM./sum_all_Cluster; %Posterior Probability

        % Maximization which is re-estimate the parameters using the current responsibilities
        for cluster=1:1:Number_of_Cluster
            mean_GMM(cluster,:)=(sum(latentVariable(:,cluster).*bimodelImage_2D))./(sum(latentVariable(:,cluster)));
            numerator = (repmat(latentVariable(:,cluster), 1, 2).* (bimodelImage_2D - mean_GMM(cluster,:)))' * ((bimodelImage_2D - mean_GMM(cluster,:)));
            denominator = sum(latentVariable(:,cluster));
            covariance{i}= numerator / denominator;
            % [~,positive_definite] = cholcov(cov);
            % if positive_definite~=0
            %    disp(strcat('Sorry, Your calculated ', num2str(i),'th',' Coveriance Matrix is not positive definite!!'));
            %    pause(2)
            %    url = 'https://pdfs.semanticscholar.org/7d4a/2da54c78cf62a2e8ea60e18cef35ab0d5e25.pdf';
            %    web(url)
            % end
            % 
            proportion_GMM(cluster)=sum(latentVariable(:,cluster))/length(bimodelImage_2D(:,1));
        end

        %% Stopping Criterion fixation
        GM=Gaussian_Mixture(bimodelImage_2D,mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
        sum_all_Cluster = sum(GM,2)+eps;
        loglikelihood_Updated=sum(log(sum_all_Cluster));

        difference_loglikelihood=loglikelihood_Updated-loglikelihood_Current;
        Store_Difference_loglikelihood(Number_of_Iteration,whichSlice)=difference_loglikelihood;

        disp(['Error--> ','Iteration = ',num2str(Number_of_Iteration),' --> ',num2str(difference_loglikelihood)]); % Display difference_loglikelihood for each Iteration

        if(abs(difference_loglikelihood)<StoppingThreshold) 
            break; 
        end
        Number_of_Iteration=Number_of_Iteration+1;
        end

        %% Display the Number of Iterations after converging to the given Stopping Threshold
        disp(['Number of Iterations Require=',num2str(Number_of_Iteration)])

        %% Pixels Classifications for the Segmenation
        PixelClassification=zeros(length(bimodelImage_2D(:,1)),1);
        for data=1:1:length(bimodelImage_2D(:,1))
          MixturePDF=Gaussian_Mixture(bimodelImage_2D(data,:),mean_GMM,covariance,proportion_GMM,Number_of_Cluster);
          Label=find(MixturePDF==max(MixturePDF));
          PixelClassification(data)=Label;
        end

        %% Segmented Image recover
        [rows,columns]=size(image_GT(:,:,1));
        PixelClassification=PixelClassification';
        FullImage=zeros(columns*rows,1);
        for i=1:1:length(OnlyBrainRegionIndex)
            FullImage(OnlyBrainRegionIndex(i))=PixelClassification(i);
        end
        img_Constructed=reshape(FullImage,[rows,columns]);

        %% Class wise Image recover. CSF=1, GM=2 and WM=3.
        img_CSF=(img_Constructed==1);
        img_GM=(img_Constructed==2);
        img_WM=(img_Constructed==3);
        Segmented_image=cat(3,img_CSF,img_GM,img_WM); %CSF=Red, GM=Green and WM=Blue
        GT_CSF=(slice_GT==1);
        GT_GM=(slice_GT==2);
        GT_WM=(slice_GT==3);
        GT_image=cat(3,GT_CSF,GT_GM,GT_WM); %CSF=Red, GM=Green and WM=Blue
        
        imshowpair(Segmented_image,GT_image,'montage')
        title('Segmented Image (Left) along with GT (Right)')

        %% DSC Calculation for each Tissue slice by slice
        DSC_CSF=Calculate_DiceCoefficient(double(slice_GT==1),img_CSF);
        DSC_CSF_Store(whichSlice)=DSC_CSF;
        DSC_GM=Calculate_DiceCoefficient(double(slice_GT==2),img_GM);
        DSC_GM_store(whichSlice)=DSC_GM;
        DSC_WM=Calculate_DiceCoefficient(double(slice_GT==3),img_WM);
        DSC_WM_store(whichSlice)=DSC_WM;
        disp(['DSC for CSF=',num2str(DSC_CSF)])
        disp(['DSC for GM=',num2str(DSC_GM)])
        disp(['DSC for WM=',num2str(DSC_WM)])
        %% Qualitative Visualization
        imshowpair(Segmented_image,GT_image,'montage')
        title(['Segmented (Left) and GT (Right) for slice = ',num2str(whichSlice)])
        xlabel(['DSC for CSF = ',num2str(DSC_CSF),',   DSC for GM = ',num2str(DSC_GM),',    and DSC for WM = ',num2str(DSC_WM)])
        pause(1)
        
    end
end
disp('-------------------DONE!!!--------------------')

%% Graphical Presentation of Error VS Iterations
figure()
plot(sum(Store_Difference_loglikelihood,2));
xlabel('Iterations');
ylabel('Error (Difference between loglikelihood)');
xlim([0 50])
title('Error vs Iteration in 2D GMM EM')
grid on;

%% Display Avg. Qauntitative Performance (DSC)
disp(['Avg. DSC for CSF=',num2str(mean(DSC_CSF_Store)),', Avg. DSC for GM=',num2str(mean(DSC_GM_store)),', and avg. DSC for WM=',num2str(mean(DSC_WM_store))])

toc;
%% Multi-variate Gaussian Mixture PDF Function
function GMM=Gaussian_Mixture(bimodelImage_2D,mean_GMM,covariance,proportion_GMM,Number_of_Cluster)
for i=1:1:Number_of_Cluster
    GMM(:,i)=proportion_GMM(i).*mvnpdf(bimodelImage_2D,mean_GMM(i,:),covariance{i});
end
end
%% --------------------- THE END-----------------------