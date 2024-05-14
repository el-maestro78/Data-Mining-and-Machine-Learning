import pandas as pd
import glob
import matplotlib.pyplot as plt
import os 
import time


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
