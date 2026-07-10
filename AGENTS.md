# AGENTS.md

本仓库是 Karpathy 路线的**实验工程**（代码 / 训练 / 调用链草稿）。  
**正式学习笔记不写在这里**，写到 Obsidian Vault。

## 双目录约定

| 角色 | 路径 |
|------|------|
| 实验工程（本仓库） | `/Users/jimmy/Projects/learning/llm-from-scratch` |
| 笔记 Vault | `/Users/jimmy/Library/Mobile Documents/iCloud~md~obsidian/Documents/Vault` |

## 写笔记时

- 主路线笔记：`Vault/01-Learning/AI-ML/Karpathy学习路线.md`
- 阶段笔记放：`Vault/01-Learning/AI-ML/` 下合适子目录（新建时保持现有命名习惯）
- 本仓库里的 `notes/`、`call-chains/`、`deploy-notes/` 只作**草稿**；有结论再整理进 Vault
- 遵守 Vault 的 Markdown / Obsidian 语法；投资相关规则与本学习线无关时可忽略

## 跨工具记忆

若需要跨会话事实，读/写 Vault 的 `memory/`（先 `memory/INDEX.md` → `memory/MEMORY.md`），不要在本仓库另起一套 memory。

## 本仓库该做什么

- 实现与修改各阶段代码（从 `01-micrograd` 起）
- 跑实验、留 README / 训练命令
- GPU：用 VS Code/Cursor 的 Colab 扩展连远端 runtime；代码仍在本机编辑

## 本仓库不要做什么

- 不要把长篇学习笔记、复盘正文只写在本仓库
- 不要 commit checkpoint / 大数据 / `.venv`
