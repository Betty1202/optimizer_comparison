# Optimizer Comparison
This repo compares common optimizers in pytorch. It compares loss, optimize time and accuracy by fine tuning BERT on GLUE-SST2 dataset for sentiment classification task. 

## Environment
I provide docker image _betty1202/pytorch:allennlp_202105_. And for this repo, you need to install python package `datasets` in the docker container.

## Usage
Run the following script to fine-tune and evaluate model.
```shell script
python main.py 
```
where, 
+ you can use `--optimizer` to choose different optimizers from _[Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD]_, and the default optimizer is `Adam`.
+ you can set epochs number by `--epoch`, and the default epoch number is `10`. 
+ you can change dataset sizr by `--dataset_size`, and the default one is `1000`, which means 1000 samples in training dataset and 100 samples in evaluation dataset.

Example for customized fine tuning:
```shell script
python main.py --optimizer SGD --epoch 5 --dataset_size 10000
```

After running the above script, you will get fine-tuning and evaluation results in _evaluation/_. Besides, _xxx_iter.xlsx_ records results from each iteration, and _xxx_epoch.xlsx_ records results from each epoch.