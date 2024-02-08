To be sorted out...

# Dataset
QQP与MRPC可用[preparation.ipynb](./preparation.ipynb)中的`Dataset`部分下载. LCQMC与BQ或取方法见相应论文.

# Rewrite Sentences
1. 为了能够批量推理, 需要将qwen目录下, `tokenization_qwen.py`文件中的
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
```
改为:
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205))) + ("<|PAD|>",)
```

2. 在[infer_qwen.py](./generation/infer_qwen.py)中设置数据集并用以下命令进行推理.
```shell
cd generation && python ./infer_qwen.py > log.log 2>&1
```

3. 用[preparation.ipynb](./preparation.ipynb)对生成结果进行预处理以及合并.