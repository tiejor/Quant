# Domain Docs

技能在探索代码库时如何读取领域文档。

## 探索前先读

- **`CONTEXT.md`** — repo 根目录的术语表
- **`docs/adr/`** — 读取与当前工作相关的架构决策记录

若这些文件不存在，静默跳过。不标记缺失，不建议创建。

## 文件结构

单 context repo：

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-python-function-factors.md
│   └── 0002-duckdb-storage.md
└── src/
```

## 使用术语表词汇

引用领域概念时，使用 `CONTEXT.md` 中定义的术语。不要漂移到术语表明确 Avoid 的同义词。

需要的概念不在术语表中时：要么你在发明项目不用的语言（重新考虑），要么确实缺条目（告知用户可用 `/grill-with-docs` 补充）。

## 标记 ADR 冲突

若输出与现有 ADR 矛盾，显式指出而非静默覆盖：

> _与 ADR-0007 矛盾 — 但值得重新讨论，因为…_
