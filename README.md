# Data Pruning for Cost Efficient Training in Tribological Analysis on Journal Bearing
MSc Computer Science (Artificial Intelligence) thesis at the University of Nottingham

## Project Overview
This paper investigates the efficacy of data pruning techniques, particularly instance reduction, to minimize processing costs during training for analyzing pressure distribution on journal-bearing systems. The study aims to explore supervised algorithms to train on a subset of prototypes instead of the entire dataset without compromising performance.

## Motivation
Existing literature primarily focuses on instance reduction techniques in classification problems, leaving a gap in regression scenarios. With the computational expense of analyzing pressure distribution, there's a need to reduce training costs. This study addresses these gaps by proposing three instance reduction algorithms tailored for regression tasks, aiming to pioneer instance reduction on journal-bearing datasets.

## Proposed Algorithms:
Multi-Task-Learning Regression Autoencoder: A novel approach clustering data points based on feature and target variable proximity.
Adaptive DBSCAN Algorithm: Customized for regression datasets' nature to dynamically adjust clustering parameters.
Modified Selesup Algorithm: Adapted from classification to regression tasks.

## Experimental Setup:
Each algorithm, alongside random sampling and full dataset usage, produces a reduced subset trained on the same neural network architecture. This allows for a comparative analysis of performance.

## Key Findings
- **Performance of Instance Reduction Algorithms**: Among the various algorithms tested, including regression autoencoder, adaptive DBSCAN, and modified Selesup, the adaptive DBSCAN algorithm demonstrated superior performance in terms of training mean absolute error, outperforming other techniques including training on the full dataset. However, it showed comparatively weaker generalization capabilities.
- **Regression Autoencoder vs Random Sampling**: The regression autoencoder outperformed random sampling, particularly at low reduction percentages. At high reduction percentages, it achieved similar performance to random sampling and even surpassed it in certain cases. Notably, training on 80% of the data, sampled using the regression autoencoder with stratified sampling, approximated training on the full dataset.

## Future work
- **Refinement of Regression Selesup Algorithm**: Given its high computational time and suboptimal performance, further modifications to the Regression Selesup algorithm are needed. Parallelizing the algorithm using big data technologies could mitigate computational costs. Additionally, adapting the algorithm to nonlinear relationships between variables, possibly by employing a small neural network instead of a linear regressor, may yield improved results.
- **Enhanced Sampling Techniques for Regression Autoencoder**: While the regression autoencoder employed a stratified sampling technique with promising results, exploring more sophisticated sampling methods tailored to the problem, such as farthest point sampling, could potentially yield even better outcomes by covering a wider range of instances.
- **Comprehensive Coverage of Data Reduction Percentages**: The research did not fully explore the range of data reduction percentages when comparing the regression autoencoder and random sampling techniques. Moreover, other algorithms mentioned were not tested across various reduction percentages. Future studies should evaluate how different algorithms perform across a wide range of data reduction percentages, providing insights into their robustness and effectiveness under varying conditions.

## Files
**Jupyter notebooks**:

Pipeline.ipynb -> Contains all the algorithms' code, training and validation for the first evaluation method (Comparison between all algorithms)

Reduction_percentage.ipynb -> Contains the second evaluation method between regression autoencoder and random sampling

Adaptive_DBSCAN.ipynb -> contains the DBSCAN code along with visualization of the algorithm's reduced subset

Regression_Selesup.ipynb -> Contains the selesup code along with visualization of the algorithm's reduced subset

Regression_Autoencoder.ipynb -> Contains the code for regression autoencoder, the training of the autoencoder, produced embeddings and extra visualizations


**CSV files**

journal_bearing.csv
Note: Access is needed from Samuel Cartwright to access this dataset
test_DBSCAN.csv -> Contains a subset of the full dataset to visualize using Adaptive_DBSCAN.ipynb

## Technologies Used
- **AWS**: For training, testing and evaluating the models
- **Sklearn**
- **Tensorflow**
- **Keras**
- **Python**
