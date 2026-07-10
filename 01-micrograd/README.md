# 01 · micrograd

目标：看懂标量自动微分、computational graph，以及 `backward()` 为什么能工作。

## 学习动作

1. 对照 [karpathy/micrograd](https://github.com/karpathy/micrograd) README / 视频
2. 在 `value.py` 里实现 `Value`（`data` / `grad` / `_prev` / `_op` / `backward`）
3. 在 `nn.py` 里搭最小 `Neuron` / `Layer` / `MLP`
4. 用 `train_toy.py` 在 toy 二分类上训练几轮
5. 笔记写回 Vault：`反向传播到底在传播什么`

## 本目录文件

| 文件 | 作用 |
|------|------|
| `value.py` | 手写标量 autograd（当前为 stub，待你填完） |
| `nn.py` | 最小神经网络 |
| `train_toy.py` | toy dataset 训练入口 |
| `notes/` | 计算图草稿（可选） |

## 运行

```bash
# 在仓库根已激活 venv 的前提下
python train_toy.py
```

上游参考可稍后 clone 到仓库外，或本阶段不放 vendor，直接手写。
