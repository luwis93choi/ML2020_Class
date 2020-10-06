#######################################
### Basic KNN Algorithm Development ###
#######################################

# Prepare dataset : [x, y, Label]
dataset = [[2.7810836,   2.550537003, 0],
	       [1.465489372, 2.362125076, 0],
	       [3.396561688, 4.400293529, 0],
	       [1.38807019,  1.850220317, 0],
	       [3.06407232,  3.005305973, 0],
	       [7.627531214, 2.759262235, 1],
	       [5.332441248, 2.088626775, 1],
	       [6.922596716, 1.77106367,  1],
	       [8.675418651,-0.242068655, 1],
	       [7.673756466, 3.508563011, 1]]

from math import sqrt

# Function for calculating euclidean distance between data from the dataset
# Dataset format : [x, y, Label]
def euclidean_distance(row1, row2):
    
    distance = 0.0
    
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i])**2
    
    return sqrt(distance)
    
row0 = [3, 3]   # Test data

for row in dataset:
    # Calculat the euclidean distance between the test data row0 and all the data in dataset
    distance = euclidean_distance(row0, row)
    print(distance)

# Locate K number of the data with the shortest distance to test data (neighbors) from the train dataset 
def get_neighbors(train, test_row, num_neighbors):
	
	distances = list()
	
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))

	# Sort the training data in ascending order of the distance to test data
	# Training data with shortest distance will be positioned in lower index
	distances.sort(key=lambda tup:tup[1])

	neighbors = list()

	# Pick K number of the data with the shortest distance to test data
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])

	return neighbors

neighbors = get_neighbors(dataset, row0, 3)	# Pick 3 neighbors to test data

for neighbor in neighbors:
	print(neighbor)

# K-Nearest Neighbor Classification
def predict_classification(train, test_row, num_neighbors):

	# (1) Select K number of neighbors (data with shortest distance) to test data
	neighbors = get_neighbors(train, test_row, num_neighbors)
	for neighbor in neighbors:
		print(neighbor)

	# (2) Make a classification prediction by conducting majority voting on neighbors' label
	output_values = [row[-1] for row in neighbors]					# Collect label of neighbors into the list
																	# Negative Index Value : Count the index from the right side

	prediction = max(set(output_values), key=output_values.count)	# Conduct the majority voting on the label of neighbors
																	# Pick the label with the highest votes among neigbhors as the prediction

	return prediction

row0 = [3, 3, 0]	# Test data

prediction = predict_classification(dataset, row0, 3)	# Classify the test data using KNN with 3 neighbors

print('Expected %d, Got %d.' % (row0[-1], prediction))	# Compare the label of test data and predcition result

######################################################
### Breast Cancer Classification Example using KNN ###
######################################################

from sklearn.datasets import load_breast_cancer		# Load breast cancer dataset

breast_cancer_data = load_breast_cancer()	# Prepare breast cancer dataset

print(breast_cancer_data.keys())			# Print the attributes of breast cancer dataset object

print(breast_cancer_data.target_names)		# Print classification target names of breast cancer dataset

import pandas as pd
df_data = pd.DataFrame(breast_cancer_data.data)			# Data matrix of breast cancer dataset
df_labels = pd.DataFrame(breast_cancer_data.target)		# Target classes of brease cancer dataset

print(df_data.head())		# Print the data matrix of the first 5 data in the breast cancer dataset
							# This shows bresat cancer dataset has 30 features for each data

print(df_labels.head())		# Print the labels of the first 5 data in the breast cancer dataset

print(df_data.describe())	# Generate the statistic description of bresat cancer dataset

# Min-Max Normalization Function
def min_max_normalize(lst):

	normalized = []

	for value in lst:
		# Normalize the values in the list using min-max normalization
		normalized_num = (value - min(lst)) / (max(lst) - min(lst))
		normalized.append(normalized_num)

	return normalized

# Conduct min-max normalization for each feature column
# Min-Max Normalization is used to prevent the negative effects of scaling (each column using different unit)
for x in range(len(df_data.columns)):

	# Each feature column list is normalized and saved back into the dataset
	df_data[x] = min_max_normalize(df_data[x])

print(df_data.describe())	# Print the statistic description of normalized breast cancer dataset

print(df_data)				# Print the normalized breast cancer dataset

# Split the dataset into training dataset and validation dataset
from sklearn.model_selection import train_test_split

# Split the dataset into 80% training dataset and 20% validation dataset
training_data, validation_data, training_labels, validation_labels = train_test_split(df_data, df_labels, test_size=0.2, random_state=100)

print(len(training_data))			# Print the number of data in training dataset
print(len(validation_data))			# Print the number of data in validation dataset
print(len(training_labels))			# Print the number of labels in training dataset
print(len(validation_labels))		# Print the number of labels in validation dataset

# KNN Classifcation on breast cancer dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)	# Prepare KNN classifier that selects 3 neighbors

classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

# Observe the changes in accuracy according to the changes in the number of neighbors
import matplotlib.pyplot as plt
k_list = range(1, 101)
accuracies = []

# Change the number of neighbors from 1 to 100
for k in k_list:
	# Conduct the classification of breast cancer using KNN with k number of neighbors
	classifier = KNeighborsClassifier(n_neighbors=k)	
	classifier.fit(training_data, training_labels)
	
	# Save the accuracy of each KNN classification case
	accuracies.append(classifier.score(validation_data, validation_labels))

# Plot the changes in accuracy according to the changes in the number of neighbors
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
