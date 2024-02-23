import psycopg2
from predict import make_predictions
import pandas as pd

# establish a connection to the database
conn = psycopg2.connect(host="DBHOST", database="DBNAME", user="DBUSER", password="DBPWD", port="DBPORT")


# Define a function to insert the prediction into the database
def insert_prediction(conn, predictions, permalinks):
    """ Insert a new prediction into the predictions table """

    # Define the INSERT statement
    insert_query = "INSERT INTO projects (score, permalink) VALUES (%s, %s)"
    
    # Insert the values into the table
    with conn.cursor() as cur:
        for i in range(len(predictions)):
            cur.execute(insert_query, (int(predictions[i]), str(permalinks.iloc[i])))
        conn.commit()
        cur.close()


# Generate test data 
data = pd.read_csv('../data/new_NFTs.csv' , index_col=None, header=0, lineterminator='\n')
# data = data.drop(['price_label', 'avg_selling_price', 'average_price'],axis=1)
# data = data.tail(50)

# Save the permalink for later
permalinks = data['permalink']
data = data.drop(['permalink'],axis=1)

# Make a prediction using the loaded model
predictions = make_predictions(data)

# Insert the prediction into the database
insert_prediction(conn, predictions, permalinks)

# Close the connection
conn.close()
