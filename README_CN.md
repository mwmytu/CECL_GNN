## Algorithmic implementation of Community Enhanced Contrastive Learning for Graph Collaborative Filtering
基于RecBole进行算法实现
### 数据集
我们采用movie-len 1m以及yelp进行实验，基于以下配置进行数据集处理:
- ml-1m
我们保留所有评分大于3的记录作为正样本
```
load_col:
  inter: [user_id, item_id, rating,timestamp]

user_inter_num_interval: "[0,inf)"
item_inter_num_interval: "[0,inf)"
val_interval:
  rating: "[3,inf)"
```
- yelp
由于yelp比较稀疏，我们保留用户交互超过15条且物品交互超过15条的用户和物品记录，同时打分必须3分及以上。并截取了
其中2013-2014年的共一年半的数据进行实验。
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

### 运行
- ml-1m
```
run_hyper.py --model=CECL --dataset=ml-1m --params_file=hyper1.test
```
- yelp
```
run_hyper.py --model=CECL --dataset=yelp --params_file=hyper1.test
```
