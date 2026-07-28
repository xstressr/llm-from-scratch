# 02 · makemore

目标：从字符级语言模型理解 embedding、MLP、batch norm、RNN、Transformer 的演进动机。

## 学习动作

1. 对照 [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) 的 makemore 系列（建议边看边改本目录代码）
2. 第 1 讲：`bigram.py`（计数表 ↔ 单层神经网络）
3. 第 2 讲：`mlp.py`（embedding + 多字符 context + MLP）
4. 第 3 讲：`batchnorm.py`（初始化 + BatchNorm + 模块化）
5. 第 4 讲：手写反向传播（Backprop Ninja）→ `backprop.py`（融合版结论）、`backprop_ninja.py`（视频 4 个练习逐步版）
6. 第 5 讲：WaveNet 风格层次融合 → `wavenet.py`
7. 每一讲一页笔记写回 Vault：`01-Learning/AI-ML/`

## 本目录文件

| 文件 | 作用 |
|------|------|
| `data/names.txt` | 人名语料（~32k） |
| `bigram.py` | 第 1 讲：count bigram + neural bigram |
| `mlp.py` | 第 2 讲：Bengio 风格 MLP |
| `batchnorm.py` | 第 3 讲：Kaiming + BatchNorm + Linear/BN/Tanh |
| `backprop.py` | 第 4 讲：融合版手写反传 vs autograd 对照（速查） |
| `backprop_ninja.py` | 第 4 讲：视频 Exercise 1-4（原子级 26 项 exact 比对 → 融合 → 手写梯度训练） |
| `wavenet.py` | 第 5 讲：模块化 → 层次融合（WaveNet 风格） |
| `plot_tanh.py` | 画 tanh / 局部梯度示意 |
| `experiments/` | 临时实验脚本 |

## 运行

```bash
# 仓库根已激活 venv，且已 pip install -e ".[torch]"
cd 02-makemore
python bigram.py
python mlp.py
python batchnorm.py
python backprop.py
python backprop_ninja.py --steps 0                     # 只做 Exercise 1-3 梯度校验
python backprop_ninja.py --exercise 4 --steps 200000   # 视频同款训练量（CPU ~50s）
python wavenet.py
```

预期：

- bigram：count NLL ≈ 2.45；neural 逼近同一量级
- mlp：test NLL ≈ 2.35（低于 bigram）；采样比 bigram 更像人名
- batchnorm：init loss ≈ log(27)≈3.30；test 与 mlp 同量级或略好
- backprop：手写梯度与 autograd `allclose=True`
- backprop_ninja：Exercise 1 的 26 项全部 `exact=True`；Exercise 2/3 是 `exact=False` + `approximate=True`（融合改变了运算顺序，属预期）；Exercise 4 跑 200k 步 val ≈ 2.17
- wavenet：阶段 1 扁平基线应贴近 batchnorm；层次融合后再比 val

## 笔记结构（写回 Vault）

- 这一讲解决什么问题？
- 旧方法哪里不够？
- 新结构改了什么？
- 损失函数和训练数据是什么？
- 有哪些可视化或 debug 方法？

上游：[karpathy/makemore](https://github.com/karpathy/makemore)
