To be sorted out...


# rewrite sentences
1. 为了能够批量推理, 需要将qwen目录下, `tokenization_qwen.py`文件中的
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
```
改为:
```python
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205))) + ("<|PAD|>",)
```

2. 在[infer_qwen.py](./generation/infer_qwen.py)中设置数据集并用以下命令进行推理
```shell
cd generation && python ./infer_qwen.py > log.log 2>&1
```