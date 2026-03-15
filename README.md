# Sampling methods comparison
Research on the influence of sample size and selection method on classification quality.
Dataset used: [Palmer Penguins Dataset for EDA](https://www.kaggle.com/datasets/satyajeetrai/palmer-penguins-dataset-for-eda)

For sample size estimation, the dataset was separated as 75% train (along which sample sizes have been estimated) and 25% test - to ensure stable testing and comparison source for all sample sizes.

Visualized Results:

<img width="975" height="321" alt="image" src="https://github.com/user-attachments/assets/d3c9bfad-f788-4cd4-b424-25f1c926cbf5" />

Fig. 1: Visualization of the work of Logistic Regression (balanced classes), K Nearest Neighbors, Random Forest models on simple random sampling – dependence of accuracy on subsample size. Confidence intervals are shown in shadows.


<img width="975" height="317" alt="image" src="https://github.com/user-attachments/assets/e91e65ad-60a5-4b02-a2cc-52d71fa13478" />

Fig. 2: Visualization of the work of Logistic Regression (balanced classes), K Nearest Neighbors, Random Forest models on Bernoulli sampling - dependence of accuracy on subsample size. Confidence intervals are shown in shadows.


<img width="975" height="325" alt="image" src="https://github.com/user-attachments/assets/6d9f0b0c-1965-42e1-927a-b58bfdf37b99" />

Fig. 3: Visualization of the performance of Logistic Regression (balanced classes), K Nearest Neighbors, Random Forest models on systematic sampling – dependence of accuracy on subsample size. Confidence intervals are shown in shadows.


<img width="975" height="321" alt="image" src="https://github.com/user-attachments/assets/dbb063fe-e619-4827-873a-1fc0ecc28ae3" />

Fig. 4: Visualization of the performance of Logistic Regression (balanced classes), K Nearest Neighbors, Random Forest models on stratified sampling – dependence of accuracy on subsample size. Confidence intervals are shown in shadows.

The study showed that the method of forming the training sample significantly affects both the average quality of the model and the stability of the results. Stratified sampling provides the lowest variance and allows achieving high F1 values ​​with smaller data volumes. Simple random and systematic methods demonstrate similar, but less stable results. Bernoulli selection is the least stable due to the additional randomness of the subsample size.

Among the models, the best resistance to downsampling of the Palmer Penguins dataset was shown by logistic regression, which indicates a clear separation of the dataset. The Random Forest model demonstrated stable, but more gradual growth dynamics. KNN turned out to be the most sensitive to the size of the training set.
Thus, for the penguin classification problem, achieving F1 ≥ 0.90 is possible even when using subsamples of a much smaller size than the full dataset, provided that stratified sampling and a sufficiently powerful model are used.

The obtained results and developments with minor edits can be applied to other datasets and, accordingly, have practical significance for assessing the minimum sufficient amount of data in applied machine learning problems.

See Zvit.pdf for the details.
