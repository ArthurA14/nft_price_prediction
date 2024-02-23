import logging
import datetime
import pandas as pd 


def data_prep():

    # observations per class
    n = 1500 

    # Set up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Add a handler to log messages to a file
    file_handler = logging.FileHandler('../logs/data_prep.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # read data
    twitter = pd.read_csv('../data/twitter_data.csv' , index_col=None, header=0, lineterminator='\n')
    logger.info("Twitter data shape: {}".format(twitter.shape))

    opensea = pd.read_csv('../data/opensea_data.csv' , index_col=0, header=0, lineterminator='\n')
    logger.info("Opensea data shape: {}".format(opensea.shape))

    # merge datasets 
    df = twitter.merge(opensea, on='permalink')
    logger.info("Data shape after merge: {}".format(df.shape))
    df.drop('permalink',axis=1,inplace=True)

    # classes balancing
    df.loc[df['price_label'] == 5] = 4
    df_subsampled0 = df[df['price_label'] == 4]
    df_subsampled1 = df[df['price_label'] == 0].sample(n=n, random_state=0)
    df_subsampled2 = df[df['price_label'] == 1].sample(n=n, random_state=0)
    df_subsampled3 = df[df['price_label'] == 2].sample(n=n, random_state=0)
    df_subsampled4 = df[df['price_label'] == 3].sample(n=n, random_state=0)

    df_concatenated = pd.concat([df_subsampled0, df_subsampled1, df_subsampled2, df_subsampled3, df_subsampled4])
    logger.info("New data shape after classes balancing step: {}".format(df_concatenated.shape))
    logger.info("New classes distribution: {}".format(df_concatenated['price_label'].value_counts()))

    # saving cleaned data to csv 
    df_concatenated.to_csv('../data/filtered_data.csv', index=False)


if __name__ == "__main__":
    data_prep()
