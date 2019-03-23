# Multi-modal-MRI-Image-Segmentation-EM-algorithm
## Written by
## Md. Kamrul Hasan
## E-mail: kamruleeekuet@gmail.com
The problem definition is to implement from scratch the algorithm of Expectation Maximization using Matlab. This algorithm has been applied on brain images (T1 and FLAIR). Three regions have to be segmented, the cerebrospinal fluid (CSF), the gray matter (GM), and the white matter (WM). All the equations used were taken from MISA course slides, see in report.

![pipeline](https://user-images.githubusercontent.com/32570071/54872737-5d43cb00-4dc9-11e9-8916-2c9254ae6666.JPG)


The work-flow that has been done is shown in Fig. \ref{fig:Pipeline} . Firstly, from both T1 and FLAIR MRI image, region of interest (ROI) has been extracted using the ground truth image. ROI selection is done by neglecting the background pixels (labeled as zeros in the ground truth). Following pseudocode used for the extraction of ROI from both T1 and FLAIR MRI image.  

\begin{figure}[h]
\centering
% Use the relevant command to insert your figure file.
% For example, with the graphicx package use
\includegraphics[width=12cm,height=17cm,keepaspectratio]{images/Pipeline_Image/pipeline.JPG}
% figure caption is below the figure
\caption{Pipeline for brain tissue segmentation using EM algorithm}
\label{fig:Pipeline}       % Give a unique label
\end{figure}

\begin{algorithm}
  \State $GT\_Image \gets \text{Read Ground Truth Image}$\;\\
  \State $Target\_Image \gets \text{Read Target Image (both T1 and FLAIR MRI)}$\;\\
\For{\texttt{<Looping for all slices>}} {
        \State \texttt{$ROI\_Index \gets find(GT\_Image(slice) \neq 0)$}\\
        \State \texttt{$ROI\_Image \gets \text{Target\_Image(ROI\_Index)}$}
      \EndFor
}
\caption{Region of Interest (ROI) selection}
\end{algorithm}

