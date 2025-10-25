---
title: 编写笔记内容
category: guide
order: 3
date: 2025-01-25
description: 掌握笔记内容的编写技巧和格式
tags: [使用指南, Markdown, 格式]
---

# 编写笔记内容

笔记系统支持完整的 Markdown 语法和多种扩展功能，让你的笔记更加丰富和易读。

## 基础语法

### 标题层级

使用 `#` 创建标题，支持 6 级标题：

```markdown
# 一级标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

**提示**：标题会自动出现在右侧的目录（TOC）中，便于导航。

### 文本格式

```markdown
**粗体文字**
*斜体文字*
~~删除线~~
`行内代码`
```

效果：
- **粗体文字**
- *斜体文字*
- ~~删除线~~
- `行内代码`

### 列表

**无序列表**：

```markdown
- 项目一
- 项目二
  - 子项目 2.1
  - 子项目 2.2
- 项目三
```

**有序列表**：

```markdown
1. 第一步
2. 第二步
3. 第三步
```

### 链接和图片

```markdown
[链接文字](https://example.com)
![图片描述](./images/example.jpg)
```

## 代码块

### 语法高亮

使用三个反引号并指定语言：

````markdown
```python
def hello_world():
    print("Hello, World!")
    return True
```
````

支持的语言包括：
- `python`, `javascript`, `typescript`
- `java`, `cpp`, `rust`, `go`
- `html`, `css`, `json`, `yaml`
- 还有更多...

### 代码示例

```javascript
// JavaScript 示例
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log(fibonacci(10)); // 55
```

```python
# Python 示例
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
```

## 数学公式

支持 KaTeX 数学公式渲染。

### 行内公式

使用单个 `$` 包裹：

```markdown
这是一个行内公式：$E = mc^2$
```

效果：这是一个行内公式：$E = mc^2$

### 块级公式

使用双 `$$` 包裹：

```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

效果：

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

### 更多公式示例

**矩阵**：

$$
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

**求和**：

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

**分数和根式**：

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

## 表格

使用 `|` 和 `-` 创建表格：

```markdown
| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 冒泡排序 | O(n²) | O(1) |
| 快速排序 | O(n log n) | O(log n) |
| 归并排序 | O(n log n) | O(n) |
```

效果：

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 冒泡排序 | O(n²) | O(1) |
| 快速排序 | O(n log n) | O(log n) |
| 归并排序 | O(n log n) | O(n) |

## 引用块

使用 `>` 创建引用：

```markdown
> 这是一段引用文字。
> 
> 可以包含多个段落。
```

效果：

> 这是一段引用文字。
> 
> 可以包含多个段落。

## 分隔线

使用三个或更多的 `-`、`*` 或 `_`：

```markdown
---
```

---

## 任务列表

```markdown
- [x] 已完成的任务
- [ ] 未完成的任务
- [ ] 另一个待办事项
```

效果：
- [x] 已完成的任务
- [ ] 未完成的任务
- [ ] 另一个待办事项

## 图片插入

### 相对路径

将图片放在笔记文件同目录或子目录：

```markdown
![示例图片](./images/example.png)
```

### 添加图片描述

```markdown
![这是图片的替代文字，当图片无法加载时显示](./demo.jpg)
```

## 最佳实践

### 1. 结构清晰

- 使用标题层级组织内容
- 一个大标题下不要放太多内容
- 善用列表和段落

### 2. 代码可读

- 为代码块指定正确的语言
- 添加必要的注释
- 保持代码格式整洁

### 3. 公式规范

- 复杂公式使用块级显示
- 为公式添加文字说明
- 使用规范的数学符号

### 4. 图文并茂

- 适当插入图片辅助理解
- 为图片添加描述性的 alt 文字
- 控制图片大小，避免过大

### 5. 链接有效

- 检查外部链接的有效性
- 内部链接使用相对路径
- 为链接添加有意义的文字

## 快捷键提示

在 VS Code 中编写 Markdown：

- `Ctrl/Cmd + B`：加粗
- `Ctrl/Cmd + I`：斜体
- `Ctrl/Cmd + Shift + V`：预览 Markdown
- `Ctrl/Cmd + K V`：在侧边预览

## 小结

现在你已经掌握了编写笔记的各种技巧！记住：

1. 合理使用标题层级
2. 善用代码高亮和数学公式
3. 保持内容结构清晰
4. 图文并茂，易于理解

在下一章，我们将介绍笔记系统的高级功能。
