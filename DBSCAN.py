# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 08:39:57 2023

@author: Adeolu
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 



#a function to read a csv file
def open_data(file_path):
    """
    This function reads a data file and returns a pandas DataFrame.
    :param file_path: path to the file
    :return: pandas DataFrame
    """
    # Use pandas to read the data file
    data = pd.read_csv(file_path)
    return data
data = open_data("Sales_Transactions_Dataset_Weekly.csv")
print(data)
normalized = data.iloc[:, 55:]



#slicing to give the output of the week columns only
df = data.iloc[:, :53]
df_1 = df.drop('Product_Code',axis=1)





#A function to create a new column for month data
def create_sum_column(df_2, col_indices):
    # create an empty list to store the sums
    sums = []
    
    # iterate over the column indices in groups of 4
    for i in range(0, len(col_indices), 4):
        # select the next 4 columns
        cols_to_sum = normalized.iloc[:, col_indices[i:i+4]]
        print('THe:',cols_to_sum)
        
        # sum the columns and append to the list
        sums.append(cols_to_sum.sum(axis=1))
    new_col = pd.Series(sums).sum()

    
    # add the new column to the DataFrame
    df_1["Month 1"] = new_col
    df_1["Month 2"] = new_col
    df_1["Month 3"] = new_col
    df_1["Month 4"] = new_col
    df_1["Month 5"] = new_col
    df_1["Month 6"] = new_col
    df_1["Month 7"] = new_col
    df_1["Month 8"] = new_col
    df_1["Month 9"] = new_col
    df_1["Month 10"] = new_col
    df_1["Month 11"] = new_col
    df_1["Month 12"] = new_col
    df_1["Month 13"] = new_col
    
    return data

# load dataset
df_2 = df_1

# call the function
col_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,\
               24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,\
                   45,46,47,48,49,50,51]
df_2 = create_sum_column(df_2, col_indices)





#A line plot to visualie the trend of the monthly data
#A line plot to visulize the trend of the weekly data
df_2 = df_1.iloc[:, 52:].sum()
print(df_2)


df_2.plot(kind='line')
plt.xlabel("Monthly sales")
plt.title('Monthly sales transaction')
plt.savefig('Monthly transaction.png', bbox_inches='tight')
plt.show()



label = 'Week 0', 'Week 10', 'Week 20', 'Week 30', 'Week 40', 'Week 50'
columns_to_sum = ['W0', 'W10', 'W20', 'W30', 'W40', 'W51']
data_sum = data[columns_to_sum].sum()

# Create a line graph
plt.plot(data_sum)
plt.ylabel('Sum')
plt.xlabel('Weekly data')
plt.title('Trend of the weekly sale Transaction')
plt.savefig('Weekly sales trend.png', bbox_inches='tight')
plt.show()




# Convert to numpy array
array = df_1.values
print(array)



#A scatter plot to show clustering of the sales data
def scatter_plot_numpy(array, x_columns, y_columns, labels = None, \
                       colors = None):
    """
    This function creates a scatter plot of multiple columns in a numpy array
    :param data: numpy array
    :param x_columns: list of column indices for x-axis
    :param y_columns: list of column indices for y-axis
    """
    if labels is None:
        labels = ['']*len(x_columns)
    if colors is None:
        colors = ['black']*len(x_columns)
    for i in range(len(x_columns)):
        x = array[:,x_columns[i]]
        y = array[:,y_columns[i]]
        plt.scatter(x, y, label = labels[i], color = colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('Scatter plot.png')
    plt.show()



scatter_plot_numpy(array, [0], [1], labels = ['df'])



#creating an object of the NearestNeighbors
#fitting the data to the object
neigh = NearestNeighbors(n_neighbors=2)
nbres= neigh.fit(array)
distance, indices= nbres.kneighbors(array)

distance=np.sort(distance, axis=0)
distance = distance[:, 1]
plt.rcParams['figure.figsize'] = (5,3)
plt.plot(distance)
plt.xlabel('Number of clusters')
plt.ylabel('distortiion')
plt.title('Elbow method showing k')
plt.show()



#K means of the data to the get the k value
sse = []
for i in range(1, 14):
    kmeans = KMeans(n_clusters= i,
    init = 'k-means++', max_iter= 300, n_init = 10)
    kmeans.fit(array)
    sse.append(kmeans.inertia_)
plt.plot(range(1,14), sse)
plt.title('The elbow showing k')
plt.xlabel("Number of clusters")
plt.ylabel('sse')
plt.savefig('Elbow.png')
plt.show()




#Performing DBSC 
arrray = StandardScaler().fit_transform(array)
dbscan = DBSCAN(eps=5, min_samples=5, metric ='euclidean',\
                algorithm ='auto').fit(arrray)
print(set(dbscan.labels_))
clusters = dbscan.fit_predict(arrray)
print(np.unique(clusters))


plt.scatter(array[clusters== -1, 0], array[clusters== -1, 1], s=10, c='green',\
            label = '-1')
plt.scatter(array[clusters== 0, 0], array[clusters== 0, 1], s=10, c='red',\
            label = '0')
plt.scatter(array[clusters== 1, 0], array[clusters== 1, 1], s=10, c='yellow',\
            label = '1')


plt.xlabel('weekly data')
plt.legend()
plt.title('DBSCAN Weekly sales transaction')
plt.savefig('DBSCAN', bbox_inches='tight')
plt.show()