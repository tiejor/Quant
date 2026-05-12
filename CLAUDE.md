# ClaudeQuant

A 股因子模型验证系统。术语表见 [CONTEXT.md](CONTEXT.md)，架构决策见 [docs/adr/](docs/adr/)。

## Agent skills

### Issue tracker

Issue 存放在 GitHub Issues（`tiejor/Quant`），通过 `gh` CLI 操作。详见 `docs/agents/issue-tracker.md`。

### Triage labels

使用 Matt Pocock 五标签体系：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。详见 `docs/agents/triage-labels.md`。

### Domain docs

单 context 布局：`CONTEXT.md` + `docs/adr/` 在 repo 根目录。详见 `docs/agents/domain.md`。
