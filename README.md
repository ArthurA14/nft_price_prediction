# NFT_price_prediction

An Open source tool to better understand the NFT market.


Us guys at Smartest Sharks, we believe that in order to make the World a better place we have to dedicate ourselves to bring our time and efforts to the Open source community. 

Open source software has greatly impacted the world by fostering collaboration and innovation. This is way we provide the Wolrd with a tool to better understand and analyze the NFT market.

"It’s free and always will be"

## Package Usage

Intall the required packages:

```bash
pip install -r requirements.txt
```

### Data preparation

#### Warning: It is necessary to get the Twitter and Opensea datasets using the scraping notebook store them in a data/ folder at the root.
#### Also: cd into the src directory.

```bash
python3  data_prep.py
```

### Training 

* lbgm model: 
```bash
python3 train.py
``` 

### Predicting on test data

```bash
python3 predict.py
```

## API Usage


```bash
gunicorn --bind 0.0.0.0:8080 app:app
```

Then provide a json payload to the endpoint URL: "http://localhost:8080/predict" (cf. code in *test_api.py*) 
  
  
