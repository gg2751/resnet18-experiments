## Set Up

To set up the experimental environment, run the following script:

```sh
sh scripts/setup.sh
```
## Running Experiments

For each of the above experiments, simply run the associated script. You can use the following script to run the first experiment (E1): 
```sh
sh scripts/E1.sh
```
You can modify the parameters and run experiments directly using `eval.py`
```sh
python eval.py --use_cuda=True --data_path='./data' --optimizer='sgd' \
                --batch_size=64 --num_workers=2 --num_gpus=2 --get_bandwidth=True
```
