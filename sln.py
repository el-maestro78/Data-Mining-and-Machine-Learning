# -*- coding: utf-8 -*-
"""sln.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IIR5sLRvCTmw4lFjF1obLqmtGdJCb0tk

# Εξόρυξη Δεδομένων και Αλγόριθμοι Μάθησης Εξαμηνιαίο Project 2023-24

<hr>

### Μέλη Ομάδας
Λογοθέτης Δημήτριος <br/>
1047106 <br/>
1047106@ac.upatras.gr <br/>

Ζαχουλίτης Κωνσταντίνος Γεώργιος <br/>
1072578 <br/>
up1072578@ac.upatras.gr

## Περιβάλλον Υλοποίησης:

Επεξεργαστής: AMD 64-bit <br>
Γλώσσα Προγραμματισμού: Python 3.12.3 <br>
IDE: PyCharm Professional <br>
Βιβλιοθήκες: NumPy, Pandas, Matplotlib, Scikit Learn, Pytorch, Pgmpy, Scipy, Seaborn <br>
Διαδικασία Εγκατάστασης Βιβλιοθηκών: pip install pandas matplotlib numpy <br>
"""

import pip
import warnings
warnings.filterwarnings("ignore", category=Warning)
def check_import(libs_list):
    for library in libs_list:
        try:
            __import__(library)
        except:
            print(f"{library} is not imported")
            pip.main(['install', library])
        else:
            print(f"{library} is imported")
libraries = ["numpy", "matplotlib", "pandas", 'scikit-learn', 'torch', 'pgmpy', 'scipy', 'seaborn']
check_import(libraries)

"""## Ερώτημα 1 - Πρώτη ανάλυση του συνόλου δεδομένων και κατάλληλες γραφικές παραστάσεις"""

# Commented out IPython magic to ensure Python compatibility.
from matplotlib import pyplot as plt
# %matplotlib inline
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
# script_directory = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join('harth')
csvs = os.listdir(CSV_PATH)

df6 = pd.read_csv(CSV_PATH + "/S008.csv")
#print(df6)
df8 = pd.read_csv(CSV_PATH + "/S009.csv")
df9 = pd.read_csv(CSV_PATH + "/S012.csv")

NaNs_df6 =  df6.isnull().sum().sort_values(ascending=False)
print(f'Df6 NaNs: {NaNs_df6}')
NaNs_df8 =  df6.isnull().sum().sort_values(ascending=False)
print(f'Df6 NaNs: {NaNs_df8}')
NaNs_df9 =  df9.isnull().sum().sort_values(ascending=False)
print(f'Df6 NaNs: {NaNs_df9}')

"""### Ενδεικτικα για συγκεκριμένα csv

#### Mean
"""

import numpy as np

plt.figure(figsize=(20, 8))

titles = ['csv6','csv8','csv9']
datasets = [df6,df8,df9]
colours = ['blue','green','red']
count = 1
for dataset in datasets:
    mean_values = []
    df = dataset
    column_names = df.columns.tolist()
    column_names.pop(0)
    column_names.pop()
    #print(column_names)
    for i in column_names:
        mean = np.mean(df[i], axis=0)
        mean_values.append(mean)

    plt.subplot(1, 3, count)
    plt.plot(column_names, mean_values, marker='o', linestyle='-', color=colours[count-1], label='Mean')
    plt.xlabel('Timestamp')
    plt.ylabel('Mean')
    plt.title(f'Mean Plot {titles[count-1]}')
    #plt.grid(True)
    count += 1


plt.show()

"""#### Median"""

plt.figure(figsize=(20, 8))

titles = ['csv6','csv8','csv9']
datasets = [df6,df8,df9]
colours = ['blue','green','red']
count = 1
for dataset in datasets:
    median_values = []
    df = dataset
    column_names = df.columns.tolist()
    column_names.pop(0)
    column_names.pop()
    #print(column_names)
    for i in column_names:
        median = np.median(df[i], axis=0)
        median_values.append(median)

    plt.subplot(1, 3, count)
    plt.plot(column_names, median_values, marker='o', linestyle='-', color=colours[count-1], label='median')
    plt.xlabel('Timestamp')
    plt.ylabel('Median')
    plt.title(f'Median Plot {titles[count-1]}')
    #plt.grid(True)
    count += 1


plt.show()

"""#### Standard Deviation"""

plt.figure(figsize=(20, 8))

titles = ['csv6','csv8','csv9']
datasets = [df6,df8,df9]
colours = ['blue','green','red']
count = 1
for dataset in datasets:
    std_values = []
    df = dataset
    column_names = df.columns.tolist()
    column_names.pop(0)
    column_names.pop()
    #print(column_names)
    for i in column_names:
        std = np.std(df[i], axis=0)  # Calculate standard deviation
        std_values.append(std)

    plt.subplot(1, 3, count)
    plt.plot(column_names, std_values, marker='o', linestyle='-', color=colours[count-1], label='Deviation')
    plt.xlabel('Timestamp')
    plt.ylabel('Deviation')
    plt.title(f'Deviation Plot {titles[count-1]}')
    #plt.grid(True)
    count += 1


plt.show()

"""#### Minimum Values"""

plt.figure(figsize=(20, 8))

titles = ['csv6','csv8','csv9']
datasets = [df6,df8,df9]
colours = ['blue','green','red']
count = 1
for dataset in datasets:
    min_values = []
    df = dataset
    column_names = df.columns.tolist()
    column_names.pop(0)
    column_names.pop()
    #print(column_names)
    for i in column_names:
        min = np.min(df[i], axis=0)
        min_values.append(min)

    plt.subplot(1, 3, count)
    plt.plot(column_names, min_values, marker='o', linestyle='-', color=colours[count-1], label='Min')
    plt.xlabel('Timestamp')
    plt.ylabel('Min Values')
    plt.title(f'Min Plot {titles[count-1]}')
    #plt.grid(True)
    count += 1


plt.show()

"""#### Maximum Values"""

plt.figure(figsize=(20, 8))

titles = ['csv6','csv8','csv9']
datasets = [df6,df8,df9]
colours = ['blue','green','red']
count = 1
for dataset in datasets:
    max_values = []
    df = dataset
    column_names = df.columns.tolist()
    column_names.pop(0)
    column_names.pop()
    #print(column_names)
    for i in column_names:
        max = np.max(df[i], axis=0)
        max_values.append(max)

    plt.subplot(1, 3, count)
    plt.plot(column_names, max_values, marker='o', linestyle='-', color=colours[count-1], label='Max')
    plt.xlabel('Timestamp')
    plt.ylabel('Max Values')
    plt.title(f'Max Plot {titles[count-1]}')
    #plt.grid(True)
    count += 1


plt.show()

"""### Για όλα τα csvs μαζί

#### Median, Mean, Standard Deviation, Min, Max
"""

funcs = [np.median,np.mean,np.std, np.min, np.max]
labels = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
fig, axs = plt.subplots(6, 5, figsize=(20,20))#, sharex=True, sharey='row')
colours = ['red', 'yellow', 'blue', 'green', 'cyan', 'magenta']
for i, label in enumerate(labels):
    for j, fun in enumerate(funcs):
        plot_values = []
        for index, file in enumerate(csvs):
            df = pd.read_csv(CSV_PATH + f"/{file}")
            df = df.reset_index(drop=True)
            res = fun(df[label], axis=0)
            plot_values.append(res)
        axs[i, j].hist(plot_values,color=colours[i], edgecolor='red') #, alpha=0.5, bins=22,
        axs[i, j].set_title(f'{fun.__name__} of {label}')

plt.tight_layout()
plt.show()

"""#### Correlation Matrix"""

import seaborn as sns
fig, axs = plt.subplots(6, 4, figsize=(30, 30))#, sharex=True, sharey='row')
labels = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']
for index, file in enumerate(csvs):
    df = pd.read_csv(CSV_PATH + f"/{file}")
    df = df.reset_index(drop=True)
    df = df[labels]
    corr_matrix = df.corr()
    ax = axs.flatten()[index]
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    #axs[index].set_title(f'{file}\'s Correlation Matrix')
    ax.set_title(f'{file}\'s Correlation Matrix')

plt.show()

"""#### Ανάλυση των Labels ανα csv"""

fig, axs = plt.subplots(6, 4, figsize=(30, 30))
for index, file in enumerate(csvs):
    df = pd.read_csv(CSV_PATH + f"/{file}")
    df = df.reset_index(drop=True)
    x_min = df.index.min()
    x_max = df.index.max()
    y_min = df['label'].min()
    y_max = df['label'].max()

    ax = axs.flatten()[index]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.scatter(df.index, df['label'])#, df['timestamp'])
    ax.scatter(df.index, df['label'])
    ax.set_title(f'{file}\'s Label Scatter')

plt.tight_layout()
plt.show()

"""#### Ανάλυση των τιμών του back_x"""

fig, axs = plt.subplots(6, 4, figsize=(30, 30))
for index, file in enumerate(csvs):
    df = pd.read_csv(CSV_PATH + f"/{file}")
    df = df.reset_index(drop=True)
    x_min = df.index.min()
    x_max = df.index.max()
    y_min = df['back_x'].min()
    y_max = df['back_x'].max()

    ax = axs.flatten()[index]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.scatter(df.index, df['label'])#, df['timestamp'])
    ax.scatter(df.index, df['back_x'])
    ax.set_title(f'{file}\'s Label Scatter')

plt.tight_layout()
plt.show()

"""#### Piechart των Labels"""

import os
import glob

def read_csv(csv_files):
    # dataX = []
    # dataY = []

    dataX = pd.DataFrame()
    dataY = pd.DataFrame()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # dataX.append(df.iloc[:, 1:7].values)
            # dataY.append(df.iloc[:, -1].values)

            dataX = pd.concat([dataX, df.iloc[:, 1:7]], ignore_index=True)
            dataY = pd.concat([dataY, df.iloc[:, -1]], ignore_index=True)

            # dataX.append(df.iloc[:,1:7].values)
            # dataX.append(df.iloc[:,-1].values)
        else:
            print("File not found")

    return dataX,dataY


# Directory containing the CSV files
directory = 'harth/'
csv_pattern = '*.csv'

# Find all CSV files in the directory
csv_files = glob.glob(directory + csv_pattern)

X,Y = read_csv(csv_files[0:5])


labels_map = {
    1: 'walking',
    2: 'running',
    3: 'shuffling',
    4: 'stairs (ascending)',
    5: 'stairs (descending)',
    6: 'standing',
    7: 'sitting',
    8: 'lying',
    13: 'cycling (sit)',
    14: 'cycling (stand)',
    130: 'cycling (sit, inactive)',
    140: 'cycling (stand, inactive)'
}

Y[0] = Y[0].map(labels_map)

value_counts = Y.value_counts()

print(value_counts)

cleaned_labels = [name[0].replace('(', '').replace(')', '').replace(',', '') for name in value_counts.index]


# # Map custom labels to value counts
# labels = []
# for val in value_counts.index.tolist():
#     if val in labels_map:
#         labels.append(labels_map[val])
#     else:
#         labels.append(str(val))  # Use the value itself as label if not found in labels_map
# counts = value_counts.tolist()


# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(value_counts, labels=cleaned_labels, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Activities')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(title="Activities", loc="upper right")
plt.show()

"""## Ερώτημα 2 - Eκπαίδευση ταξινομητών και σύγκριση των μοντέλων

### 2α) Neural Networks
"""

import  pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=Warning)
torch.__version__

df6 = pd.read_csv(CSV_PATH + "/S006.csv") # πολλα απο 1-7
df9 = pd.read_csv(CSV_PATH + "/S009.csv") # πολλα δεδομενα cycling
csvs2 = csvs[:]
if "S006.csv" in csvs2:
    csvs2.pop(0)
    csvs2.pop(1)
ultimate_df = pd.concat([df6, df9])
for file in csvs2:
    new_df = pd.read_csv(CSV_PATH + f"/{file}")
    ultimate_df = pd.concat([ultimate_df , new_df], axis=0)

try:
    ultimate_df.drop(columns='index', inplace=True)
except KeyError as e:
    print(e)
print(ultimate_df.head())

numeric_columns = df6.select_dtypes(include=['number']).columns
print(numeric_columns)

"""Ορισμός του Μοντέλου μας"""

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class HarthClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(HarthClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        hidden_half = hidden_size // 2
        self.fc2 = nn.Linear(hidden_size, hidden_half)
        self.fc3 = nn.Linear(hidden_half, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        #return nn.functional.softmax(x, dim=1)
        return x


def train(num_epochs, model, trainloader, val_loader, lr):
    predictions = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in trainloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions.append(outputs.detach().numpy())

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    predictions = np.concatenate(predictions, axis=0)
    return predictions

"""Ετοιμάζουμε τα datasets"""

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split

mapping = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:4,
    6:5,
    7:6,
    8:7,
    13: 8,
    14: 9,
    130: 10,
    140: 11
}

# Array splitting
array = ultimate_df[numeric_columns].values
train_size = int(0.7 * len(array))  # 70% of the data for training
val_size = int(0.15 * len(array))  # 15% of the data for validation
test_size = len(array) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(array, [train_size, val_size, test_size])


# Train set
train_dataset = np.array(train_dataset)
X_train = train_dataset[:, :-1]
y_train = train_dataset[:, -1]
X_train_T = torch.tensor(X_train, dtype=torch.float32)
y_train_T = torch.tensor([mapping.get(label, label.item()) for label in y_train], dtype=torch.long)
train_dataset_T = TensorDataset(X_train_T, y_train_T)
trainloader = DataLoader(train_dataset_T, batch_size=128, shuffle=True)

# Validation set
val_dataset = np.array(val_dataset)
X_val = val_dataset[:, :-1]
y_val = val_dataset[:, -1]
X_val_T = torch.tensor(X_val, dtype=torch.float32)
y_val_T = torch.tensor([mapping.get(label, label.item()) for label in y_val], dtype=torch.long)
val_dataset_T = TensorDataset(X_val_T, y_val_T)
val_loader = DataLoader(val_dataset_T, batch_size=128, shuffle=False)

# Test Set
test_dataset = np.array(test_dataset)
X_test = test_dataset[:, :-1]
y_test = test_dataset[:, -1]
X_test_T = torch.tensor(X_test, dtype=torch.float32)
y_test_T = torch.tensor(y_test, dtype=torch.long)
test_dataset_T = TensorDataset(X_test_T, y_test_T)
test_loader = DataLoader(test_dataset_T, batch_size=128, shuffle=False)

"""Εκπαιδεύουμε το Μοντέλο"""

model = HarthClassifier(input_size=6, hidden_size=10, output_size=12,  dropout_prob=0.5)
try:
    pred = train(num_epochs=50, model=model, trainloader=trainloader , val_loader=val_loader, lr=0.001)
except IndexError as e:
    print(e)

"""Αξιολόγηση του μοντέλου και Προβλέψεις σε άλλα csvs

"""

def evaluate(df, model):
    numeric_columns = df.select_dtypes(include=['number']).columns
    try:
        df.drop(columns=['Index'], inplace=True)
    except:
        pass
    try:
        df.drop(columns=['index'], inplace=True)
    except:
        pass
    #print(df.columns)
    new_array = df[numeric_columns].values
    X_new_array = new_array[:, :-1]
    X_new = torch.tensor(X_new_array, dtype=torch.float32)
    y_new = new_array[:, -1].astype(int)
    with torch.no_grad():
        model.eval()
        predictions = model(X_new)

    predictions_array = predictions.numpy()
    predicted_labels = np.argmax(predictions_array, axis=1)
    reverse_mapping = {v: k for k, v in mapping.items()}
    original_labels = np.array([reverse_mapping[label] if label in reverse_mapping.keys() else label for label in predicted_labels])
    accuracy = (y_new == original_labels).mean()
    return accuracy

for file in csvs:
    df = pd.read_csv(CSV_PATH + f"/{file}")
    try:
        accuracy = evaluate(df=df, model=model)
        print(f'Accuracy for {file}: {accuracy:.4f}')
    except:
        print(f'For file {file} failed')

"""Δοκιμαζουμε και το Test Dataset

"""

total_pred = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        predictions = model(inputs)
        total_pred .append(predictions.detach().numpy())

predictions_array = np.concatenate(total_pred, axis=0)
predicted_labels = np.argmax(predictions_array, axis=1)
reverse_mapping = {v: k for k, v in mapping.items()}
original_labels = np.array([reverse_mapping[label] if label in reverse_mapping.keys() else label for label in predicted_labels])
accuracy = (y_test == original_labels).mean()

print(f'Test Set Accuracy: {accuracy*100:.2f}%')

from sklearn.metrics import f1_score, precision_score, recall_score

def metrics(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    return val_loss, f1, precision, recall

val_loss, f1, precision, recall = metrics(model, val_loader)
print(f'Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

"""### 2β) Random Forests

"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import ShuffleSplit
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

"""### 2γ) Bayesian Networks

#### Λύση με Bayesian Network μέσω PGMPY
"""

import pgmpy
import os
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=Warning)
pgmpy.__version__

"""Προετοιμασία των datasets"""

CSV_PATH = os.path.join('harth')
df6 = pd.read_csv(CSV_PATH + "/S006.csv")
df9 = pd.read_csv(CSV_PATH + "/S009.csv")
df20 = pd.read_csv(CSV_PATH + "/S020.csv")
df28 = pd.read_csv(CSV_PATH + "/S028.csv")
df15 = pd.read_csv(CSV_PATH + "/S015.csv")
df27 = pd.read_csv(CSV_PATH + "/S027.csv")
df29 = pd.read_csv(CSV_PATH + "/S029.csv")
ultimate_df = pd.concat([df6, df9, df20, df28, df15, df27, df29])

data_bayes_train = ultimate_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']]
data_bayes_train.dropna()
data_bayes_train.drop_duplicates()

from sklearn.model_selection import train_test_split

train_bayes, test_bayes = train_test_split(data_bayes_train, test_size=0.3, random_state=42)
train_bayes = train_bayes.sample(1000, random_state=42)
test_bayes = test_bayes.sample(1000, random_state=42)

"""Βοηθητικές Συναρτήσεις"""

from pgmpy.inference import VariableElimination, BeliefPropagation
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_bn(df, bn):
    predictor = VariableElimination(bn)
    #predictor = BeliefPropagation(bn)
    #predictor.calibrate()
    results = []
    classes = {}
    i = 0
    for index, c in enumerate(df['label']):
        if c not in classes.values():
            classes[i] = c
            i+=1
    #evidence_variables = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    evidence_variables = ['thigh_z', 'thigh_x']
    no_ev_count = 0
    for index, row in df.iterrows():
        evidence = row[evidence_variables].to_dict()
        try:
            predicted_distribution = predictor.query(variables=['label'], evidence=evidence, show_progress=False)
        except Exception as e:
            no_ev_count+=1
            predicted_distribution = predictor.query(variables=['label'], evidence=None, show_progress=False)
        predicted_label = predicted_distribution.values.argmax()
        results.append(classes[predicted_label])
        #results.append(predicted_label)

    precision = precision_score(df['label'], results, average='weighted')
    recall = recall_score(df['label'], results, average='weighted')
    f1 = f1_score(df['label'], results, average='weighted')

    print(F"Precision: {precision:.4f}, Recall: {recall:.4f}, f1: {f1:.4f}")
    print(no_ev_count/len(df))
    return precision, recall, f1

def ensure_consistent_states(data, state_names):
    for column in data.columns:
        if column in state_names:
            data[column] = pd.Categorical(data[column], categories=state_names[column])
        else:
            state_names[column] = data[column].unique().tolist()
    return data, state_names

"""Δημιουργούμε το Μοντέλο"""

from pgmpy.models import BayesianNetwork
bayesian = BayesianNetwork([
    ('back_x', 'label'),
    ('back_y', 'label'),
    ('back_z', 'label'),
    ('thigh_x', 'label'),
    ('thigh_y', 'label'),
    ('thigh_z', 'label')
    ])

"""Εκπαίδευση του Μοντέλου"""

batch_size = 13
state_names = {}
initial_data = train_bayes[:batch_size]
subsequent_batches = [train_bayes[i:i+batch_size] for i in range(batch_size, len(train_bayes), batch_size)]
initial_data, state_names = ensure_consistent_states(initial_data, state_names)
bayesian.fit(initial_data, state_names=state_names)
n_prev_samples = len(initial_data)


for batch in subsequent_batches:
    batch, state_names = ensure_consistent_states(batch, state_names)
    try:
        bayesian.fit_update(batch, n_prev_samples=n_prev_samples)
        n_prev_samples += len(batch)
    except ValueError as e:
        print(f"Error updating with batch: {e}")
bayesian.check_model()

"""Αξιολόγηση

Σε άγνωστα δεδομένα
"""

prec_m, recl_m, f1_m = evaluate_bn(test_bayes, bayesian)

"""Σε γνωστά δεδομένα"""

prec_m, recl_m, f1_m = evaluate_bn(train_bayes, bayesian)

"""#### Λύση με Gauss Naive Bayes Classifier (Όχι Bayesian Network)

Προετοιμάζουμε το dataset
"""

import os
CSV_PATH = os.path.join('harth')
df6 = pd.read_csv(CSV_PATH + "/S006.csv")
df20 = pd.read_csv(CSV_PATH + "/S020.csv")
df28 = pd.read_csv(CSV_PATH + "/S028.csv")
df15 = pd.read_csv(CSV_PATH + "/S015.csv")
ultimate_df = pd.concat([df6, df15])

data_bayes_train = ultimate_df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label']]
data_bayes_train.dropna()
data_bayes_train.drop_duplicates()

from sklearn.model_selection import train_test_split
# Χωριζουμε σε train & test
train_bayes, test_bayes = train_test_split(data_bayes_train, test_size=0.3, random_state=42)

"""Δημιουργούμε το Μοντέλο"""

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

y_train = train_bayes['label']
X_train = train_bayes[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
X_train.dropna()
y_train.dropna()

# scaling επειδή το μοντέλο δεν λειτουργεί με αρνητικές τιμές
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
log_X_Train= np.log(X_train + 100)
log_X_Train = np.nan_to_num(log_X_Train, nan=0)


clf = GaussianNB().fit(log_X_Train, y_train)

#Testing
test_no_lab = test_bayes.drop(columns='label')
test_no_lab = scaler.fit_transform(test_no_lab)
test_no_lab= np.log(test_no_lab + 100)
test_no_lab = np.nan_to_num(test_no_lab, nan=0)
predicted_classes = clf.predict(test_no_lab)

"""Αξιολόγηση"""

from sklearn.metrics import precision_score, recall_score, f1_score


precision = precision_score(test_bayes['label'], predicted_classes, average='weighted')
recall = recall_score(test_bayes['label'], predicted_classes, average='weighted')
f1 = f1_score(test_bayes['label'], predicted_classes, average='weighted')

print(F"Precision: {precision:.4f}, Recall: {recall:.4f}, f1: {f1:.4f}")

"""## Ερώτημα 3 - Συσταδοποίηση και μετασχηματισμός του συνόλου δεδομένων"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.utils import resample
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

"""Cleaning Csvs from unwanted columns"""

def read_csv(csv_files):

    dataX = pd.DataFrame()
    dataY = pd.DataFrame()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            #dataX = pd.concat([dataX, df.iloc[:, 1:7]], ignore_index=True)
            #dataY = pd.concat([dataY, df.iloc[:, -1]], ignore_index=True)
            dataX = pd.concat([dataX, df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]], ignore_index=True)
            dataY = pd.concat([dataY, df.iloc[:, -1]], ignore_index=True)

        else:
            print("File not found")

    return dataX,dataY


# Directory containing the CSV files
directory = 'harth/'
csv_pattern = '*.csv'

# Find all CSV files in the directory
csv_files = glob.glob(directory + csv_pattern)

#['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']

"""Ετοιμάζουμε το dataset κάνοντας τις κατάλληλες αλλαγές"""

X, Y = read_csv(csv_files[0:23])

X = X.sample(n=20000, random_state=42)
# Transforming the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(X)
normalized_Y = scaler.fit_transform(Y)

"""#### Συσταδοποίηση μέσω K-means

Εύρεση Optimal k δοκιμάζοντας πιθανές τιμές και αξιολογώντας μέσω Sihlouette Score
"""

best_silhouette = 0.0
best_overall_silhouette = 0.0
optimal_K = 0
best_overall_k = 0

for i in range(5): # Για δοκιμή διαφορετικών samples, με διαφορετικά k.
    sample_size = min(5000, len(normalized_features)//2)
    sampled_features = resample(normalized_features, n_samples=sample_size, random_state=42+(i*i))
    iter = 30
    for k in range(2,iter):
        kmeans = KMeans(n_clusters=k, random_state=42, max_iter=50)
        clusters = kmeans.fit_predict(sampled_features)
        silhouette = silhouette_score(sampled_features , clusters)
        print(f'K={k}/{iter-1}. Silhouette score: {silhouette:.5f}')
        if silhouette > best_silhouette:
            optimal_K = k
            best_silhouette = silhouette
    print('-------'*10)
    if best_silhouette > best_overall_silhouette:
        best_overall_silhouette = best_silhouette
        best_overall_k = optimal_K
        #optimal_sample_kmeans = sampled_features

print(f'Silhouette score for Optimal k={best_overall_k}: {best_overall_silhouette:.5f}')

optimal_K_chi = 0
best_chi = 0.0

for k in range(2, iter):
    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100)
    clusters = kmeans.fit_predict(sampled_features) # Του από πάνω κελιού
    calinski_harabasz = calinski_harabasz_score(sampled_features, clusters)
    if calinski_harabasz > best_chi:
        optimal_K_chi  = k
        best_chi = calinski_harabasz

print(f'Calinski-Harabasz score for Optimal k={optimal_K_chi}: {best_chi:.5f}')

possible_ks = [optimal_K_chi, best_overall_k]
for i in range(5):
    sample_size = min(10000, len(normalized_features)//2)
    sampled_features = resample(normalized_features, n_samples=sample_size, random_state=42+(i*i))
    print(f'Iter {i+1}/5')
    for k in possible_ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(sampled_features)
        sil_score = silhouette_score(sampled_features , clusters) # Του από πάνω κελιού
        chi_score = calinski_harabasz_score(sampled_features, clusters)
        print(f'Silhouette score for k={k}: {sil_score:.5f}')
        print(f'Calinski-Harabasz score for k={k}: {chi_score:.5f}')
    print('-'*20)

optimal_K = best_overall_k
kmeans = KMeans(n_clusters=best_overall_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features)
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={best_overall_k}: {kmeans_silhouette:.5f}')
print('-------'*10)
chi_score = calinski_harabasz_score(normalized_features, kmeans_clusters)
print(f'Calinski-Harabasz score for k={best_overall_k}: {chi_score:.5f}')

X['cluster'] = kmeans_clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
reduced_data = pd.DataFrame(reduced_features, columns=['clusters', 'X'])
reduced_data['cluster'] = kmeans_clusters
# Creating the scatter plot

plt.scatter(reduced_data['cluster'], reduced_data['X'], c=reduced_data['cluster'], cmap='viridis')
plt.xlabel('Clusters')
plt.ylabel('X')
plt.title('K-means Clustering')
plt.colorbar(label='Kmean Clusters')
plt.show()

"""Προσπάθεια εύρεσης του βέλτιστου k μέσω Elbow Method"""

k_values = range(1, 200)

# Store the inertia for each k
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

k_values = range(1, 5000, 100)

# Store the inertia for each k
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

K_elbow = 25
kmeans = KMeans(n_clusters=K_elbow, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features)
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={K_elbow}: {kmeans_silhouette:.5f}')
chi_score = calinski_harabasz_score(normalized_features, kmeans_clusters)
print(f'Calinski-Harabasz score for k={K_elbow}: {chi_score:.5f}')
print('-------'*10)
X['cluster'] = kmeans_clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
reduced_data = pd.DataFrame(reduced_features, columns=['clusters', 'X'])
reduced_data['cluster'] = kmeans_clusters
# Creating the scatter plot

plt.scatter(reduced_data['cluster'], reduced_data['X'], c=reduced_data['cluster'], cmap='viridis')
plt.xlabel('Clusters')
plt.ylabel('X')
plt.title('K-means Clustering')
plt.colorbar(label='Kmean Clusters')
plt.show()

"""#### Συσταδοποίηση μέσω Hierarchical Clustering

Εύρεση Optimal n - αριθμού clusters μέσα από Dendrogram
"""

linked = linkage(normalized_features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True)
plt.show()

from scipy.cluster.hierarchy import fcluster
cutoff_height = 25
clusters = fcluster(linked, t=cutoff_height , criterion='distance')

cutoff = len(set(clusters))
print(f"The number of clusters at height {cutoff_height } is {cutoff}")

hc = AgglomerativeClustering(n_clusters=cutoff)
hc_clusters= hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={cutoff}: {hc_silhouette:.5f}')
chi_score = calinski_harabasz_score(normalized_features, hc_clusters)
print(f'Calinski-Harabasz score for n={cutoff}: {chi_score:.5f}')
print('-------'*10)

X['cluster'] = hc_clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
reduced_data = pd.DataFrame(reduced_features, columns=['clusters', 'X'])
reduced_data['cluster'] = hc_clusters
plt.scatter(reduced_data['cluster'], reduced_data['X'], c=reduced_data['cluster'], cmap='viridis')
plt.xlabel('Clusters')
plt.ylabel('X')
plt.title('Hierarchical Clustering')
plt.colorbar(label='Hierarchical Clusters')
plt.show()

"""Εύρεση Optimal n - δοκιμάζοντας πιθανές τιμές και αξιολογώντας μέσω Sihlouette Score"""

best_overall_silhouette_hc = 0.0
optimal_N = 0
best_silhouette_hc = 0.0
best_overall_n = 0
for _ in range(5):
    sample_size = min(5000, len(normalized_features)//2)
    sampled_features = resample(normalized_features, n_samples=sample_size, random_state=42)
    iter = 30
    for n in range(2, iter):
        hc = AgglomerativeClustering(n_clusters=n)
        clusters = hc.fit_predict(sampled_features)
        silhouette = silhouette_score(sampled_features, clusters)
        print(f'N={n}/{iter-1}. Silhouette score: {silhouette:.5f}')
        if silhouette > best_silhouette_hc:
            optimal_N = n
            best_silhouette_hc = silhouette
    print('-------'*10)
    if best_silhouette_hc > best_overall_silhouette_hc:
        best_overall_silhouette_hc = best_silhouette_hc
        best_overall_n = optimal_N
        optimal_sample_hc = sampled_features
print(f'Silhouette score for Optimal n={best_overall_n}: {best_overall_silhouette_hc:.5f}')

optimal_N_chi = 0
best_chi = 0.0

for n in range(2, iter):
    hc = AgglomerativeClustering(n_clusters=n)
    clusters = hc.fit_predict(sampled_features) # Του από πάνω κελιού
    calinski_harabasz = calinski_harabasz_score(sampled_features, clusters)
    if calinski_harabasz > best_chi:
        optimal_N_chi  = n
        best_chi = calinski_harabasz

print(f'Calinski-Harabasz score for Optimal k={optimal_N_chi}: {best_chi:.5f}')

possible_ns = [optimal_N_chi, best_overall_n]
for i in range(5):
    sample_size = min(10000, len(normalized_features)//2)
    sampled_features = resample(normalized_features, n_samples=sample_size, random_state=42+(i*i))
    print(f'Iter {i+1}/5')
    for n in possible_ns:
        hc = AgglomerativeClustering(n_clusters=n)
        clusters = hc.fit_predict(sampled_features)
        sil_score = silhouette_score(sampled_features , clusters) # Του από πάνω κελιού
        chi_score = calinski_harabasz_score(sampled_features, clusters)
        print(f'Silhouette score for k={n}: {sil_score:.5f}')
        print(f'Calinski-Harabasz score for k={n}: {chi_score:.5f}')
    print('-'*20)

hc = AgglomerativeClustering(n_clusters=best_overall_n)
hc_clusters= hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={best_overall_n}: {hc_silhouette:.5f}')
chi_score = calinski_harabasz_score(normalized_features, hc_clusters)
print(f'Calinski-Harabasz score for n={best_overall_n}: {chi_score:.5f}')
print('-------'*10)
X['cluster'] = hc_clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(normalized_features)
reduced_data = pd.DataFrame(reduced_features, columns=['clusters', 'X'])
reduced_data['cluster'] = hc_clusters
plt.scatter(reduced_data['cluster'], reduced_data['X'], c=reduced_data['cluster'], cmap='viridis')
plt.xlabel('Clusters')
plt.ylabel('X')
plt.title('Hierarchical Clustering')
plt.colorbar(label='Hierarchical Clusters')
plt.show()

"""#### Σύγκριση"""

kmeans = KMeans(n_clusters=best_overall_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features )
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={best_overall_k}: {kmeans_silhouette:.5f}')
print('-'*20)
hc = AgglomerativeClustering(n_clusters=best_overall_n)
hc_clusters = hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={best_overall_n}: {hc_silhouette:.5f}')

kmeans = KMeans(n_clusters=optimal_K_chi, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features )
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={optimal_K_chi}: {kmeans_silhouette:.5f}')
print('-'*20)
hc = AgglomerativeClustering(n_clusters=optimal_N_chi)
hc_clusters = hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={optimal_N_chi}: {hc_silhouette:.5f}')

big_k = big_n = 2000
kmeans = KMeans(n_clusters=big_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features )
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={big_k}: {kmeans_silhouette:.5f}')
print('-'*20)
hc = AgglomerativeClustering(n_clusters=big_n)
hc_clusters = hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={big_n}: {hc_silhouette:.5f}')

very_big_k = very_big_n = 10000
kmeans = KMeans(n_clusters=very_big_k, random_state=42)
kmeans_clusters = kmeans.fit_predict(normalized_features )
kmeans_silhouette = silhouette_score(normalized_features , kmeans_clusters)
print(f'Silhouette score for Optimal k={very_big_k}: {kmeans_silhouette:.5f}')
print('-'*20)
hc = AgglomerativeClustering(n_clusters=very_big_n)
hc_clusters = hc.fit_predict(normalized_features)
hc_silhouette = silhouette_score(normalized_features , hc_clusters)
print(f'Silhouette score for Optimal n={very_big_n}: {hc_silhouette:.5f}')

sns.pairplot(X, hue='cluster', diag_kind='hist')
plt.show()