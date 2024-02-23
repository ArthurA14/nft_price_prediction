import requests
import json
import pandas as pd

# Define the endpoint URL
url = "http://localhost:8080/predict"

# Generate test data 
data = pd.read_csv('data/new_NFTs.csv' , index_col=None, header=0, lineterminator='\n')
# data = data.drop(['price_label', 'avg_selling_price', 'average_price'],axis=1)
data = data.tail(50)

# Convert the dataframe to a list of dictionaries
data_json = data.to_dict('records')

# Send a request for each data point in the list
for i, item in enumerate(data_json):
    
    # Send the request with the JSON payload
    response = requests.post(url, json=item)

    # Print the prediction
    print(response.text)
