---
title: 浅谈A*算法
published: 2024-11-17 22:09:36
tags: ["算法", "A星", "路径搜索"] # 将 tags 修改为数组，并添加了一些相关的标签作为示例
category: 算法
---

<!--more-->

## 1. 算法思想

    A* 算法是一种启发式搜索算法，常用于在图中寻找从起始节点到目标节点的最短路径。它结合了 Dijkstra 算法（保证能找到最优解的广度优先搜索拓展）的特点以及启发式函数来引导搜索方向，以减少搜索空间。
    算法维护两个集合，一个是已探索的节点集合，另一个是待探索的节点集合（通常用优先队列实现，按照节点的评估函数值排序）。每个节点都有一个评估函数 `f(n)`，它由两部分组成：从起始节点到当前节点 `n` 的实际代价 `g(n)`（类似 Dijkstra 算法中的距离度量），以及从当前节点 `n` 到目标节点的预估代价 `h(n)`（启发式函数），即 `f(n) = g(n) + h(n)`。
    在搜索过程中，每次从待探索集合中取出 `f(n)` 值最小的节点进行扩展，直到找到目标节点或者待探索集合为空。启发式函数 `h(n)` 需要满足一定的条件（可采纳性，即 `h(n)` 始终小于等于节点 `n` 到目标节点的实际最短距离），这样才能保证 A* 算法最终找到的是最优解。

## 2. 代码示例

（Python 代码，以简单的二维网格地图为例，寻找起点到终点的最短路径，地图中 0 表示可通行，1 表示障碍物）

    ```python

    import heapq

    # 定义四个方向的移动向量（上下左右）
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    def heuristic(a, b):
        """
        启发式函数，这里使用曼哈顿距离（在二维网格中常用），计算两点之间的预估距离
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(grid, start, goal):
        """
        A* 算法实现
        """
        rows = len(grid)
        cols = len(grid[0])
        # 初始化 g 值，初始设为无穷大
        g_scores = [[float('inf')] * cols for _ in range(rows)]
        g_scores[start[0]][start[1]] = 0
        # 初始化 f 值，初始设为无穷大
        f_scores = [[float('inf')] * cols for _ in range(rows)]
        f_scores[start[0]][start[1]] = heuristic(start, goal)
        # 优先队列，存储待探索的节点，元素为 (f值, (x坐标, y坐标))
        open_set = [(f_scores[start[0]][start[1]], start)]
        came_from = {}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            for i in range(4):
                new_x = current[0] + dx[i]
                new_y = current[1] + dy[i]
                if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 0:
                    tentative_g_score = g_scores[current[0]][current[1]] + 1
                    if tentative_g_score < g_scores[new_x][new_y]:
                        came_from[(new_x, new_y)] = current
                        g_scores[new_x][new_y] = tentative_g_score
                        f_scores[new_x][new_y] = tentative_g_score + heuristic((new_x, new_y), goal)
                        heapq.heappush(open_set, (f_scores[new_x][new_y], (new_x, new_y)))
        return None
    ```

你可以使用以下方式调用这个函数：

    ```python

    # 示例地图
    grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    start = (0, 0)
    goal = (3, 3)
    path = a_star(grid, start, goal)
    print(path)

    ```

## 3. 复杂度分析

- **时间复杂度**：在最坏情况下（比如启发式函数不够好，接近均匀搜索整个图空间时），A* 算法的时间复杂度与图中节点数 `n` 和边数 `m` 相关，类似于 Dijkstra 算法，时间复杂度为 $O((n + m) \log n)$，其中 $\log n$ 来自于优先队列操作的复杂度。但如果启发式函数非常好，能有效引导搜索方向，实际运行时间可以大大优于这个复杂度，接近最优解所在路径长度的线性复杂度。
- **空间复杂度**：需要存储每个节点的 `g` 值、`f` 值以及记录节点的前驱等信息，在最坏情况下，空间复杂度为 $O(n)$，`n` 为图中节点的数量，主要取决于节点数以及需要记录的辅助信息占用的空间。

## 4. 正确性证明（基于可采纳性）

- 假设存在一条从起始节点 `s` 到目标节点 `t` 的最优路径 `P`，其长度为 `L`。我们要证明 A* 算法一定能找到这条最优路径。
- 由于启发式函数 `h(n)` 满足可采纳性，即对于任意节点 `n`，`h(n)` 始终小于等于节点 `n` 到目标节点的实际最短距离。
- 当 A* 算法扩展节点时，它总是选择 `f(n)` 值最小的节点进行扩展，其中 `f(n) = g(n) + h(n)`。因为 `h(n)` 不会高估到目标的距离，所以如果某个节点 `n` 在最优路径 `P` 上，那么最终 `f(n)` 值会小于等于 `L`（最优路径长度），且沿着最优路径上的节点依次被扩展，直到到达目标节点 `t`。
- 而且，不会出现因为优先选择了其他非最优路径上的节点而错过最优路径的情况，因为那些非最优路径上的节点最终计算出的 `f` 值必然大于最优路径对应的 `f` 值（由于 `h` 函数的可采纳性以及实际代价 `g` 的累积），所以 A* 算法最终一定会沿着最优路径扩展到目标节点，从而找到最优解。
