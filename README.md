# Multi-modal-MRI-Image-Segmentation-EM-algorithm
## Written by
## Md. Kamrul Hasan
## E-mail: kamruleeekuet@gmail.com
The problem definition is to implement from scratch the algorithm of Expectation Maximization using Matlab. This algorithm has been applied to brain images (T1 and FLAIR). Three regions have to be segmented: the cerebrospinal fluid (CSF), the gray matter (GM), and the white matter (WM). All the equations used were taken from the MISA course slides, see the report.

The workflow that has been done is shown in Figure below. Firstly, from both the T1 and FLAIR MRI images, a region of interest (ROI) has been extracted using the ground truth image. ROI selection is done by neglecting the background pixels (labeled as zeros in the ground truth). 

![pipeline](https://user-images.githubusercontent.com/32570071/54872737-5d43cb00-4dc9-11e9-8916-2c9254ae6666.JPG)

After selecting ROI, a feature vector with NxD dimensions has been created, where N indicates the number of pixels inside the ROI and D indicates the dimension of the feature vector, which is 2 (T_1 weighted and FLAIR weighted MRI). In 3D implementation, N is the total number of pixels inside the ROI for all slices, while in 2D implementation, N is the total number of pixels inside the ROI for one slice inside the loop of slice-by-slice processing.

k-means clustering has been used to get the initial parameters, i.e., the mean of each cluster, co-variance matrices, and cluster priorities. For different runs, k-means assigns different cluster labels randomly. But, in the ground truth, cluster labels are fixed, i.e., CSF = 1, GM = 2, and WM = 3.






