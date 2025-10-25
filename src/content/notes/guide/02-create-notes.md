---
title: 创建新笔记
category: guide
order: 2
date: 2025-01-25
description: 学习如何创建和组织笔记内容
tags: [使用指南, 创建笔记]
---

# 创建新笔记

本章将详细介绍如何在博客中创建和组织笔记。

## 文件结构

笔记存放在 `src/content/notes/` 目录下，按课程分类：

```
src/content/notes/
├── 课程名称1/
│   ├── 01-第一章.md
│   ├── 02-第二章.md
│   └── ...
├── 课程名称2/
│   └── ...
└── ...
```

## 创建步骤

### 1. 创建课程文件夹

首先，在 `notes` 目录下创建一个新的文件夹，用课程名称命名：

```bash
mkdir src/content/notes/算法进阶
```

**命名建议**：
- 使用清晰、简洁的名称
- 可以使用中文或英文
- 避免特殊字符

### 2. 创建笔记文件

在课程文件夹中创建 Markdown 文件：

```bash
# 建议使用序号前缀，便于排序
touch src/content/notes/算法进阶/01-动态规划基础.md
```

### 3. 编写 Front Matter

每个笔记文件必须包含 Front Matter（元数据）：

```markdown
---
title: 动态规划基础          # 笔记标题
category: 算法进阶          # 课程分类（文件夹名）
order: 1                    # 显示顺序
date: 2025-01-25           # 创建日期
description: 介绍动态规划的基本概念和解题思路  # 简短描述
tags: [动态规划, 算法]      # 标签（可选）
draft: false                # 是否为草稿（可选，默认 false）
---

# 这里开始写笔记内容
```

## Front Matter 字段说明

### 必填字段

- **title**：笔记标题，显示在页面顶部
- **category**：课程分类，必须与文件夹名称一致
- **order**：排序序号，数字越小越靠前

### 可选字段

- **date**：创建或更新日期
- **description**：简短描述，用于搜索和预览
- **tags**：标签数组，便于分类检索
- **draft**：设为 `true` 则不会在网站上显示
- **image**：封面图片路径（相对于笔记文件）

## 示例：完整笔记

```markdown
---
title: 二分查找详解
category: 算法基础
order: 3
date: 2025-01-25
description: 深入理解二分查找算法的原理和应用
tags: [查找算法, 二分查找]
---

# 二分查找详解

## 算法原理

二分查找是一种高效的查找算法，时间复杂度为 O(log n)。

### 基本思想

在有序数组中，通过比较中间元素来缩小查找范围...

## 代码实现

\`\`\`python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
\`\`\`

## 复杂度分析

- **时间复杂度**：O(log n)
- **空间复杂度**：O(1)

## 应用场景

1. 有序数组中查找元素
2. 查找插入位置
3. 旋转数组问题
...
```

## 注意事项

### Category 命名

`category` 字段的值必须与文件夹名完全一致：

```markdown
✅ 正确
文件夹：notes/算法基础/
Front Matter: category: 算法基础

❌ 错误
文件夹：notes/算法基础/
Front Matter: category: algorithm  # 不一致！
```

### Order 排序

`order` 值决定笔记的显示顺序：

```markdown
01-introduction.md    → order: 1
02-basics.md          → order: 2
03-advanced.md        → order: 3
```

### 文件名规范

虽然文件名不影响显示，但建议：
- 使用序号前缀（如 `01-`、`02-`）
- 使用有意义的英文名或拼音
- 避免空格和特殊字符

## 下一步

现在你已经学会了如何创建笔记，接下来我们将学习如何编写丰富的笔记内容。
