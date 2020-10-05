
################################################
### Titanic Classification Example using KNN ###
################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

titanic_dataset = pd.read_csv('./train.csv')

print(titanic_dataset)

print(titanic_dataset.shape)

print('[Pre-Processing] : Before pre-processing NaN values of Age column')
print(titanic_dataset.isnull().sum())

titanic_dataset['Age'].fillna(titanic_dataset.groupby('Sex')['Age'].transform('mean'), inplace=True)

print('[Pre-Processing] : Replace NaN values of Age column as average age of gender')
print(titanic_dataset.isnull().sum())

print('[Pre-Processing] : Drop Cabin and Embarked Features')
titanic_dataset = titanic_dataset.drop(['Cabin', 'Embarked'], axis=1)
print(titanic_dataset.isnull().sum())

print('[Pre-Processing] : Label text-based features into numeric labels')
sex_mapping = {'male' : 0, 'female' : 1}
titanic_dataset['Sex'] = titanic_dataset['Sex'].map(sex_mapping)

print('[Pre-Processing] : Use Pclass, Sex, Embarked as dataset features, and Survived as Label / Filter out the rest')
dataset_features = titanic_dataset.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Parch', 'Fare', 'SibSp'], axis=1)
dataset_labels = titanic_dataset['Survived']

print('[Pre-Processed Titanic Dataset]')
print(dataset_features)

print('[Pre-Processing] : Split the dataset into 70% training dataset and 30% validation dataset')
train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels, test_size=0.3, random_state=32)

k_list = range(1, 101)
accuracy = []

for k in k_list:

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_features, train_labels)
    accuracy.append(classifier.score(test_features, test_labels))

# Plot the changes in accuracy according to the changes in the number of neighbors
plt.plot(k_list, accuracy, 'bo-')
plt.xlabel('Number of Neigbhors in KNN')
plt.ylabel('Validation Accuracy')
plt.title('Titanic KNN Classifier Accuracy')
plt.grid()
plt.xticks(k_list, fontsize='x-small')
plt.tight_layout()
plt.show()
