## Algorithmic implementation of Community Enhanced Contrastive Learning for Graph Collaborative Filtering
Algorithm implementation based on RecBole
### Dataset
We used movie-len 1m as well as yelp for our experiments and processed the dataset based on the following configuration.
- ml-1m
We kept all records with ratings greater than 3 as a positive sample.
```
load_col:
  inter: [user_id, item_id, rating,timestamp]

user_inter_num_interval: "[0,inf)"
item_inter_num_interval: "[0,inf)"
val_interval:
  rating: "[3,inf)"
```
- yelp
Since yelp is sparse, we keep records of users and items with more than 15 user interactions and more than 15 item interactions, while scoring must be 3 and above. And intercepted
which a total of one and a half years of data from 2013-2014 for the experiment.
```
load_col:
  inter: [user_id, item_id, rating, timestamp]
ITEM_ID_FIELD: item_id
RATING_FIELD: rating

user_inter_num_interval: "[15,inf)"
item_inter_num_interval: "[15,inf)"
val_interval:
  timestamp: "[1356969600, 1404144000)"
  rating: "[3,inf)"
```

## Run
- ml-1m
```
run_hyper.py --model=CECL --dataset=ml-1m --params_file=hyper1.test
```
- yelp
```
run_hyper.py --model=CECL --dataset=yelp --params_file=hyper1.test
```
