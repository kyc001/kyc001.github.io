---
title: 高级功能
category: guide
order: 4
date: 2025-01-25
description: 探索笔记系统的高级功能和技巧
tags: [使用指南, 高级功能]
---

# 高级功能

掌握这些高级功能，让你的笔记更加强大和专业。

## 目录（TOC）配置

### 自动生成目录

笔记页面会自动在右侧显示目录（需要 2xl 及以上屏幕），包含：

- 文章的所有标题
- 当前阅读位置高亮
- 点击快速跳转

### 控制目录深度

在 `src/config.ts` 中配置：

```typescript
export const siteConfig = {
  // ...
  toc: {
    enable: true,     // 是否启用 TOC
    depth: 2,         // 显示的标题层级深度（1-3）
  },
};
```

**depth 说明**：
- `1`：仅显示一级标题（`#`）
- `2`：显示一、二级标题（`#`、`##`）
- `3`：显示一、二、三级标题（`#`、`##`、`###`）

### 目录最佳实践

1. **标题层级合理**：避免跳级使用标题
   ```markdown
   ✅ 正确
   # 一级标题
   ## 二级标题
   ### 三级标题
   
   ❌ 错误
   # 一级标题
   ### 三级标题（跳过了二级）
   ```

2. **标题简洁明了**：TOC 中会显示标题全文
3. **避免过深嵌套**：建议不超过 3 级标题

## 草稿功能

### 什么是草稿

将笔记标记为草稿，暂时不在网站上显示：

```markdown
---
title: 未完成的笔记
category: guide
order: 99
draft: true        # 标记为草稿
---
```

### 使用场景

- 笔记尚未完成
- 内容需要进一步修订
- 等待资料补充

### 开发模式预览

在开发环境（`npm run dev`）中，草稿会正常显示，便于预览和编辑。

## 封面图片

### 添加封面

在 Front Matter 中指定封面图片：

```markdown
---
title: 我的笔记
category: guide
order: 1
image: ./cover.jpg    # 相对于笔记文件的路径
---
```

### 图片位置

将图片放在笔记文件同目录或子目录：

```
notes/guide/
  ├── 01-intro.md
  ├── cover.jpg          # 封面图片
  └── images/
      └── banner.png     # 也可以放在子目录
```

引用方式：
```markdown
image: ./cover.jpg              # 同目录
image: ./images/banner.png      # 子目录
```

### 图片要求

- **格式**：JPG、PNG、WebP 等
- **尺寸**：建议宽度 1200px 以上
- **大小**：控制在 500KB 以内
- **比例**：推荐 16:9 或 2:1

## 上一篇/下一篇导航

### 自动生成

系统会根据 `order` 字段自动生成同课程内的导航：

```markdown
[← 上一篇: 创建新笔记]  [下一篇: 高级功能 →]
```

### 导航逻辑

1. 按 `order` 升序排列
2. 自动跳过草稿（`draft: true`）
3. 仅在同一课程内导航

### 优化建议

合理设置 `order` 值，使导航顺序符合学习逻辑：

```markdown
01-basics.md        → order: 1   # 基础概念
02-installation.md  → order: 2   # 安装配置
03-first-app.md     → order: 3   # 第一个应用
04-advanced.md      → order: 4   # 进阶内容
```

## 标签系统

### 添加标签

在 Front Matter 中使用数组格式：

```markdown
---
tags: [算法, 动态规划, 面试题]
---
```

### 标签显示

标签会显示在笔记的元信息区域，便于分类和检索。

### 标签建议

- 使用 2-5 个标签
- 标签简洁明了
- 建立标签体系，保持一致性

示例标签体系：
```
技术类：前端、后端、算法、数据库
难度：入门、进阶、高级
类型：教程、笔记、总结
```

## 数学公式高级用法

### 复杂公式示例

**动态规划状态转移方程**：

$$
dp[i][j] = \begin{cases}
dp[i-1][j-1] & \text{if } s[i] = t[j] \\
\min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 & \text{otherwise}
\end{cases}
$$

**矩阵运算**：

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

**概率公式**：

$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

### 常用符号

```latex
希腊字母：\alpha, \beta, \gamma, \theta, \lambda, \pi, \sigma
运算符：\sum, \int, \prod, \lim, \frac{}{}
关系符：\leq, \geq, \neq, \approx, \equiv
集合符：\in, \subset, \cup, \cap, \emptyset
箭头：\rightarrow, \Rightarrow, \leftarrow, \Leftarrow
```

## SEO 优化

### 元数据设置

确保填写完整的元数据：

```markdown
---
title: 具体的、有描述性的标题
description: 详细描述笔记内容，50-160 字符
tags: [相关, 关键词]
---
```

### JSON-LD 结构化数据

系统自动为每个笔记生成 JSON-LD 数据，提升搜索引擎收录：

- `@type`: "LearningResource"
- 标题、描述、关键词
- 发布日期
- 课程信息

## 性能优化

### 图片优化

1. **压缩图片**：使用工具压缩后再上传
2. **选择合适格式**：
   - 照片：JPG
   - 图标、图表：PNG
   - 现代浏览器：WebP
3. **懒加载**：系统自动实现

### 代码块优化

- 避免过长的代码块（超过 100 行）
- 可以分段展示或提供链接

## 快捷工作流

### 1. 批量创建笔记

使用脚本快速创建笔记模板：

```bash
# 创建笔记目录
mkdir -p src/content/notes/新课程

# 批量创建文件
for i in {1..5}; do
  echo "---
title: 第 $i 章
category: 新课程
order: $i
---

# 第 $i 章
" > "src/content/notes/新课程/0$i-chapter.md"
done
```

### 2. VS Code 代码片段

创建 Markdown 代码片段，快速插入 Front Matter：

```json
{
  "Note Front Matter": {
    "prefix": "note",
    "body": [
      "---",
      "title: ${1:标题}",
      "category: ${2:分类}",
      "order: ${3:1}",
      "date: ${CURRENT_YEAR}-${CURRENT_MONTH}-${CURRENT_DATE}",
      "description: ${4:描述}",
      "tags: [${5:标签}]",
      "---",
      "",
      "# ${1:标题}",
      "",
      "$0"
    ],
    "description": "笔记 Front Matter 模板"
  }
}
```

### 3. Git 工作流

```bash
# 创建新笔记分支
git checkout -b note/新主题

# 提交笔记
git add src/content/notes/
git commit -m "Add: 新主题笔记"

# 合并到主分支
git checkout main
git merge note/新主题
```

## 故障排除

### 笔记不显示

检查清单：
- [ ] Front Matter 格式正确
- [ ] `category` 与文件夹名一致
- [ ] `draft` 未设为 `true`
- [ ] `order` 字段存在

### TOC 不显示

- [ ] 检查 `config.ts` 中 `toc.enable` 是否为 `true`
- [ ] 确认笔记中有标题
- [ ] 浏览器窗口宽度足够（2xl+）

### 公式渲染错误

- [ ] 检查 LaTeX 语法
- [ ] 特殊字符是否转义
- [ ] 公式是否闭合

## 小结

通过本章学习，你已经掌握了：

- ✅ TOC 配置和优化
- ✅ 草稿和封面图片
- ✅ 导航和标签系统
- ✅ 数学公式高级用法
- ✅ SEO 和性能优化
- ✅ 高效工作流

现在，你可以充分发挥笔记系统的威力，创建出色的学习笔记了！
