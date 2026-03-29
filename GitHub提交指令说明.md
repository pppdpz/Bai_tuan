# GitHub 提交指令说明（RoboVerse）

## 1. 目标与思路

本文用于说明在 `/home/e0/RoboVerse` 中，如何规范地完成：

1. 本地改动检查
2. 提交（commit）
3. 推送到 GitHub（push）
4. 子模块（`pyroki`）的提交与同步

---

## 2. 首次配置（只需一次）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

如果当前目录还不是 Git 仓库：

```bash
cd /home/e0/RoboVerse
git init -b main
git remote add origin https://github.com/pppdpz/Bai_tuan.git
```

如果 `origin` 已存在，改成新地址：

```bash
git remote set-url origin https://github.com/pppdpz/Bai_tuan.git
```

---

## 3. 每次提交的标准流程

### 3.1 查看改动

```bash
cd /home/e0/RoboVerse
git status
git diff
```

### 3.2 添加改动到暂存区

全部添加：

```bash
git add -A
```

按文件添加（更安全）：

```bash
git add 路径/文件1 路径/文件2
```

### 3.3 提交

```bash
git commit -m "feat: 简要描述本次改动"
```

提交信息建议：

1. `feat:` 新功能
2. `fix:` 修复问题
3. `docs:` 文档修改
4. `refactor:` 重构
5. `chore:` 杂项维护

### 3.4 推送到 GitHub

首次推送当前分支：

```bash
git push -u origin main
```

后续推送：

```bash
git push
```

---

## 4. 推荐分支工作流（避免直接改 main）

```bash
git switch -c feat/xxx
git add -A
git commit -m "feat: xxx"
git push -u origin feat/xxx
```

然后在 GitHub 上发起 PR 合并到 `main`。

---

## 5. `pyroki` 作为子模块时的操作

你当前计划是保留 `pyroki` 独立历史，推荐放在 `third_party/pyroki`。

### 5.1 在子模块里提交

```bash
cd /home/e0/RoboVerse/third_party/pyroki
git status
git add -A
git commit -m "fix: 子模块改动说明"
git push
```

### 5.2 回到主仓库，记录子模块新指针并提交

```bash
cd /home/e0/RoboVerse
git add third_party/pyroki .gitmodules
git commit -m "chore: update pyroki submodule pointer"
git push
```

说明：主仓库不会存子模块全部代码，只会记录子模块“指向的提交号”。

---

## 6. 大文件注意事项（GitHub 常见失败点）

GitHub 普通 Git 不接受超过 100MB 的单文件。  
若推送失败，常见方案：

1. 忽略大文件（写入 `.gitignore`）
2. 使用 Git LFS 管理大文件

示例（忽略某类大文件）：

```bash
echo "roboverse_data/assets/EmbodiedGenData/**/gs_model.ply" >> .gitignore
git add .gitignore
git commit -m "chore: ignore large local assets"
git push
```

---

## 7. 常用排错命令

查看远程地址：

```bash
git remote -v
```

查看最近提交：

```bash
git log --oneline --decorate -n 20
```

取消暂存（不删文件）：

```bash
git restore --staged 路径/文件
```

查看未推送提交：

```bash
git log --oneline origin/main..HEAD
```

---

## 8. 最小可执行模板

```bash
cd /home/e0/RoboVerse
git status
git add -A
git commit -m "chore: update project files"
git push
```

如果是新分支：

```bash
git switch -c feat/my-change
git add -A
git commit -m "feat: my change"
git push -u origin feat/my-change
```
