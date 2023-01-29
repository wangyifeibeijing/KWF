## running environment

* torch and dgl latest

## running procedure

* Download Data folder and pretrain folder
* unzip them to current folder
* cd to Model folder and run

## run

```bash
python -u main.py --model_type baseline --gpu_id 1 --ue_lambda 0.1 --idf_sampling 1 --dataset last-fm --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1

python -u main.py --model_type baseline --dataset movie-lens --gpu_id 1 --ue_lambda 0.4 --idf_sampling 1 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1

python -u main.py --model_type baseline --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --dataset amazon-book --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1

python -u main.py --model_type baseline --gpu_id 0 --ue_lambda 0.1 --idf_sampling 1 --dataset yelp2018 --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 3000 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 8192 --sprate 1
```