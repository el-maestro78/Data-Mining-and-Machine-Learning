""" 
Data columns
1. timestamp: date and time of recorded sample
2. back_x: acceleration of back sensor in x-direction (down) in the unit g
3. back_y: acceleration of back sensor in y-direction (left) in the unit g
4. back_z: acceleration of back sensor in z-direction (forward) in the unit g
5. thigh_x: acceleration of thigh sensor in x-direction (down) in the unit g
6. thigh_y: acceleration of thigh sensor in y-direction (right) in the unit g
7. thigh_z: acceleration of thigh sensor in z-direction (backward) in the unit g
8. label: annotated activity code

Hierarchical Clustering
 """

""" 
Exercise values: label
1: walking	
2: running	
3: shuffling
4: stairs (ascending)	
5: stairs (descending)	
6: standing	
7: sitting	
8: lying	
13: cycling (sit)	
14: cycling (stand)	
130: cycling (sit, inactive)
140: cycling (stand, inactive)
 """
 
 
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os 
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt


def read_csv(csv_files):
    
    dataX = pd.DataFrame()
    dataY = pd.DataFrame()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            dataX = pd.concat([dataX, df.iloc[:, 1:7]], ignore_index=True)
            dataY = pd.concat([dataY, df.iloc[:, -1]], ignore_index=True)
            
        else:
            print("File not found")
        
    return dataX,dataY


# Directory containing the CSV files
directory = 'harth/'
csv_pattern = '*.csv'

# Find all CSV files in the directory
csv_files = glob.glob(directory + csv_pattern)


X, Y = read_csv(csv_files[0:1])

# Timing start
start_time = time.time()

# Create the Random Forest Classifier and spliting the data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
ss = ShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=0)

# Training the model and outputing crucial data
for train_index, test_index in ss.split(X, Y):
    
    X_values = X.values
    Y_values = Y.values.ravel()
    
    X_train, X_test = X_values[train_index], X_values[test_index]
    y_train, y_test = Y_values[train_index], Y_values[test_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Output the classification result matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 
    
    # Output the confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  
    
    # Output the accuracy of the model
    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
    
    # prediction data can be imported from another file, formatted the same as Data of X
    print(clf.predict(X_train))  
    print(clf.score(X_test, y_test))

# Calculating feature importance
importances = clf.feature_importances_  
print(importances)

# Timing end
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Drawing a bar chart
plt.barh(range(len(importances)), importances)
plt.title("Feature Importances")
feature_names = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
# Adding feature names
plt.yticks(range(len(importances)), feature_names)

plt.show()

