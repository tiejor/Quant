# Issue tracker: GitHub

Issues 和 PRD 存放在 GitHub Issues。所有操作使用 `gh` CLI。

## 操作约定

- **创建 Issue**：`gh issue create --repo tiejor/Quant --title "..." --body "..."`。多行内容用 heredoc。
- **读取 Issue**：`gh issue view <number> --comments`
- **列出 Issue**：`gh issue list --repo tiejor/Quant --state open --json number,title,body,labels`
- **评论 Issue**：`gh issue comment <number> --body "..."`
- **加/删标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭 Issue**：`gh issue close <number> --comment "..."`

`gh` 在 repo 内运行时自动推断仓库，但显式指定 `--repo tiejor/Quant` 更可靠。

## "发布到 Issue 追踪器"

创建 GitHub Issue。

## "获取相关工单"

执行 `gh issue view <number> --comments`。
