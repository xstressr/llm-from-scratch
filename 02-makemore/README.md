# 02 · makemore

目标：从字符级语言模型理解 embedding、MLP、batch norm、RNN、Transformer 的演进动机。

## 学习动作

1. 对照 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 的 makemore 系列（建议边看边改本目录代码）
2. 第 1 讲：`bigram.py`（计数表 ↔ 单层神经网络）
3. 第 2 讲：`mlp.py`（embedding + 多字符 context + MLP）
4. 第 3 讲：`batchnorm.py`（初始化 + BatchNorm + 模块化）
5. 第 4 讲：手写反向传播（Backprop Ninja）→ 计划 `backprop.py`
6. 第 5 讲：WaveNet 风格更深结构
7. 每一讲一页笔记写回 Vault：`01-Learning/AI-ML/`

## 本目录文件

| 文件 | 作用 |
|------|------|
| `data/names.txt` | 人名语料（~32k） |
| `bigram.py` | 第 1 讲：count bigram + neural bigram |
| `mlp.py` | 第 2 讲：Bengio 风格 MLP |
| `batchnorm.py` | 第 3 讲：Kaiming + BatchNorm + Linear/BN/Tanh |
| `plot_tanh.py` | 画 tanh / 局部梯度示意 |
| `experiments/` | 临时实验脚本 |

## 运行

```bash
# 仓库根已激活 venv，且已 pip install -e ".[torch]"
cd 02-makemore
python bigram.py
python mlp.py
python batchnorm.py
```

预期：

- bigram：count NLL ≈ 2.45；neural 逼近同一量级
- mlp：test NLL ≈ 2.35（低于 bigram）；采样比 bigram 更像人名
- batchnorm：init loss ≈ log(27)≈3.30；test 与 mlp 同量级或略好

## 笔记结构（写回 Vault）

- 这一讲解决什么问题？
- 旧方法哪里不够？
- 新结构改了什么？
- 损失函数和训练数据是什么？
- 有哪些可视化或 debug 方法？

上游：[karpathy/makemore](https://github.com/karpathy/makemore)
