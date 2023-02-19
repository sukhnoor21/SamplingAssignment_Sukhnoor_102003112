import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import Counter
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans



data=pd.read_csv("Creditcard_data.csv")

data.shape

#Distribution of Two classes
data.Class.value_counts()

#Highly Imblanced dataset
# 0--> Legit Transactions
# 1--> Fraudlent Transactions

# Separating Legit and Fraudlent Transactions
X = data.drop('Class',axis=1)
Y = data['Class']

print(X.shape)
print(Y.shape)

from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
rus = RandomOverSampler(random_state=0)
X_train_os, Y_train_os = rus.fit_resample(X,Y)

print("Before Over Sampling :" , Counter(Y))
print("After Over Sampling :" , Counter(Y_train_os))

balanced_df = pd.concat([X_train_os, Y_train_os], axis=1)
balanced_df.to_csv('balanced_dataset.csv', index=False)
print("Balanced Dataset Created...")



# Calculation of sample size using sample size detection formula
p = np.sum(Y_train_os) / len(Y_train_os)

# Set the confidence level and margin of error
confidence_level = input("Enter the Confidence Level(in %) :")  
confidence_level = float(confidence_level)/100

alpha = 1-confidence_level
print("The Margin of Error :",alpha)

z_score = norm.ppf(1-alpha/2)
print("Z-Score is :",z_score)

n = int(np.ceil((z_score**2 * p * (1-p)) / (alpha**2)))
print("Sample Size is :",n)

sample_datasets = []

# 1. Simple Random Sampling
sample1 = balanced_df.sample(n, replace=False)
sample_datasets.append(sample1)

# 2. Stratified Sampling
sample4=balanced_df.groupby('Class',group_keys=False).apply(lambda x: x.sample(frac=.2523))
sample_datasets.append(sample4)

# 3. systematic Sampling
sampling_interval = int(len(balanced_df) / n) # Sample every 10th row

# Create a list of indices to sample
indices = np.arange(start=0, stop=len(balanced_df), step=sampling_interval)[:n]

# Sample the dataset using the indices
sample0 = balanced_df.iloc[indices]
sample_datasets.append(sample0)

# 4. Cluster Sampling
from sklearn.cluster import KMeans
# Separate the data into two clusters based on the class column
kmeans = KMeans(n_clusters=2, random_state=0).fit(balanced_df.iloc[:, :-1])
clusters = kmeans.predict(balanced_df.iloc[:, :-1])
balanced_df['cluster'] = clusters

# Calculate the proportion of each cluster in the data
proportions = balanced_df['cluster'].value_counts(normalize=True)

# Set the desired sample size and calculate the number of samples to take from each cluster
desired_sample_size = n
sample_sizes = np.round(proportions * n).astype(int)

# Initialize an empty list to store the sampled data
sample3 = []

# Iterate over each cluster and take a random sample of the appropriate size
for cluster, size in sample_sizes.iteritems():
    cluster_data = balanced_df[balanced_df['cluster'] == cluster]
    sample = cluster_data.sample(n=size, random_state=0)
    sample3.append(sample)

# Concatenate the sampled data into a single DataFrame
sample3 = pd.concat(sample3)

# Remove the cluster column from the sampled data
sample3 = sample3.drop('cluster', axis=1)
sample_datasets.append(sample3)

# 5. convenience Sampling
# Separate the 0 and 1 class observations into two separate dataframes and Conducting convenience sampling by selecting n/2 observations from each class
zeros_df = balanced_df[balanced_df['Class'] == 0].sample(int(n/2), random_state=1)
ones_df = balanced_df[balanced_df['Class'] == 1].sample(int(n/2), random_state=1)

# Combine the sampled dataframes into a new balanced dataset
sample1 = pd.concat([zeros_df, ones_df])
sample_datasets.append(sample1)

# Save the sample datasets to separate CSV files
for i, sample in enumerate(sample_datasets):
    sample.to_csv(f'sample_dataset_{i}.csv', index=False)

models = [
    Pipeline([('scaler', StandardScaler()),('lr', LogisticRegression(max_iter=1000))]),
    SVC(), GaussianNB(), KNeighborsClassifier(),RandomForestClassifier()]

# Define a list of model names for the table
model_names = ['Logistic Regression', 'SVC', 'Naive bayes', 'KNN', 'Random forest']

# Define a table to store the results
results_table = pd.DataFrame(columns=['Dataset', *model_names])

# Loop over each sample dataset and each model to compute accuracy
for i, sample in enumerate(sample_datasets):
    X = sample.iloc[:, :-1]
    y = sample.iloc[:, -1]
    row = {'Dataset': f'Sampling {i+1}'}
    for j, model in enumerate(models):
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        row[model_names[j]] = f'{accuracy:.3f}'
    results_table = results_table.append(row, ignore_index=True)

# Transpose the table so that the model names are in the first column and dataset names are in the top row
results_table = results_table.set_index('Dataset').T.rename_axis('Model', axis=0)

# Print the results table
print(results_table)
results_table.to_csv(f'Final_Solution.csv', index=False)