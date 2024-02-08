To be sorted out...

# Dataset
QQP and MRPC can be downloaded in the 'Dataset' section of [preparation.ipynb](./preparation.ipynb). LCQMC and BQ can be obtained in corresponding papers.
# Rewrite Sentences
1. In order to batch infer, you need to modify the following code in the 'tokenization_qwen.py' file of qwen model directory
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
```
to
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205))) + ("<|PAD|>",)
```

2. Set up the dataset in [infer_qwen.py](./generation/infer_qwen.py) and infer with the following command.
```shell
cd generation && python ./infer_qwen.py > log.log 2>&1
```

3. The generation results can be preprocessed and combined using [prepreparation.ipynb](./preparation.ipynb).
# Traing
Using `*.sh` scripts to train.