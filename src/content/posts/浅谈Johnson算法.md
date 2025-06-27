---
title: 浅谈 Johnson 算法
abbrlink: 38955
date: 2024-11-17 22:09:36
tags:
categories:
---

<!-- # Johnson 算法 -->

## 1. 算法思想

Johnson 算法用于解决带权有向图中所有节点对之间的最短路径问题，它结合了 Bellman-Ford 算法和 Dijkstra 算法的优点。
首先，通过添加一个虚拟节点 `s` 到图中，将所有边的权重进行重新赋值（重赋权技巧），使得图中不存在负权环的同时，利用 Bellman-Ford 算法计算出从虚拟节点 `s` 到图中每个节点的最短距离 `h(v)`。
然后，对于图中的每一对节点 `u` 和 `v`，将原边权重 `w(u, v)` 替换为新的权重 `w'(u, v) = w(u, v) + h(u) - h(v)`，这样处理后可以保证新的权重是非负的（通过利用最短距离的性质）。
最后，针对每一个节点作为源点，使用 Dijkstra 算法来计算经过重赋权后的图中到其他节点的最短路径，再根据重赋权的关系将计算出的距离转换回原图中的最短路径距离。

## 2. 代码示例

（Python 代码，以下是简化实现，假设输入的图用邻接矩阵表示，图中不存在负权环，节点编号从 0 开始）

```python

    import math

    def bellman_ford(graph, num_vertices):
        """
        Bellman-Ford 算法，用于计算从虚拟源点到各节点的最短距离
        """
        # 初始化距离数组，初始设为无穷大
        dist = [math.inf] * num_vertices
        dist[0] = 0
        for _ in range(num_vertices - 1):
            for u in range(num_vertices):
                for v in range(num_vertices):
                    if graph[u][v]!= math.inf:
                        if dist[u] + graph[u][v] < dist[v]:
                            dist[v] = dist[u] + graph[u][v]
        return dist

    def dijkstra(graph, source, num_vertices):
        """
        Dijkstra 算法，用于计算从给定源点到其他节点的最短路径
        """
        # 初始化距离数组，初始设为无穷大
        dist = [math.inf] * num_vertices
        dist[source] = 0
        visited = [false] * num_vertices
        for _ in range(num_vertices):
            min_dist = math.inf
            min_index = -1
            for v in range(num_vertices):
                if not visited[v] and dist[v] < min_dist:
                    min_dist = dist[v]
                    min_index = v
            visited[min_index] = true
            for v in range(num_vertices):
                if graph[min_index][v]!= math.inf and dist[min_index] + graph[min_index][v] < dist[v]:
                    dist[v] = dist[min_index] + graph[min_index][v]
        return dist

    def johnson(graph):
        """
        Johnson 算法实现
        """
        num_vertices = len(graph)
        # 添加虚拟源点，构建新的图
        new_graph = [[math.inf] * (num_vertices + 1) for _ in range(num_vertices + 1)]
        for i in range(num_vertices):
            for j in range(num_vertices):
                new_graph[i][j] = graph[i][j]
        # 运行 Bellman-Ford 算法
        h = bellman_ford(new_graph, num_vertices + 1)[0:num_vertices]
        # 重赋权边
        new_weight_graph = [[math.inf] * num_vertices for _ in range(num_vertices)]
        for u in range(num_vertices):
        for v in range(num_vertices):
            if graph[u][v]!= math.inf:
                new_weight_graph[u][v] = graph[u][v] + h[u] - h[v]
        # 对每个节点作为源点运行 Dijkstra 算法，存储所有节点对最短路径结果
        all_pairs_shortest_paths = []
        for source in range(num_vertices):
            dist = dijkstra(new_weight_graph, source, num_vertices)
            # 还原原权重下的最短路径距离
            for v in range(num_vertices):
                if dist[v]!= math.inf:
                    dist[v] = dist[v] - h[source] + h[v]
            all_pairs_shortest_paths.append(dist)
        return all_pairs_shortest_paths
```

你可以使用以下方式调用这个函数：

```python

    # 示例图，用邻接矩阵表示，无穷大表示无边相连
    graph = [
        [0, 3, math.inf, 7],
        [8, 0, 2, math.inf],
        [5, math.inf, 0, 1],
        [2, math.inf, math.inf, 0]
    ]
    shortest_paths = johnson(graph)
    print(shortest_paths)

```

## 3. 复杂度分析

- **时间复杂度**：

  - 第一步 Bellman-Ford 算法的时间复杂度为 $O(V^3)$，其中 `V` 是图中节点的数量，因为它要对每条边进行 `V - 1` 次松弛操作，每次松弛操作遍历所有边（在邻接矩阵表示下），时间复杂度为 $O(V^2)$，整体就是 $O(V^3)$。

  - 第二步对每个节点运行 Dijkstra 算法，若使用二叉堆实现优先队列的 Dijkstra 算法，每次时间复杂度为 $O((V + E) \log V)$，`E` 是边数，总共运行 `V` 次，这部分时间复杂度就是 $O(V (V + E) \log V)$。在稀疏图（`E` 接近 `V`）情况下，时间复杂度约为 $O(V^2 \log V)$，在稠密图（`E` 接近 $V^2$）情况下，时间复杂度约为 $O(V^3 \log V)$。所以 Johnson 算法总的时间复杂度在最坏情况下为 $O(V^3 \log V)$。

- **空间复杂度**：需要存储图的相关信息（如邻接矩阵等）、距离数组等，在 Bellman-Ford 和 Dijkstra 算法执行过程中还需要一些临时空间，总体空间复杂度主要取决于图的表示方式以及节点数量，一般来说在 $O(V^2)$ 级别（使用邻接矩阵表示图时），`V` 为节点数量。

## 4. 正确性证明

- **重赋权后不存在负权边**：
根据 Bellman-Ford 算法计算出的 `h(v)` 是从虚拟节点 `s` 到节点 `v` 的最短距离。对于图中任意边 `(u, v)`，有 `w'(u, v) = w(u, v) + h(u) - h(v)`，根据最短路径的三角不等式（从 `s` 到 `v` 的最短路径长度不会大于从 `s` 经过 `u` 再到 `v` 的路径长度），可得 `h(v) ≤ h(u) + w(u, v)`，移项后就是 `w'(u, v) = w(u, v) + h(u) - h(v) ≥ 0`，所以重赋权后的边权重都是非负的。

- **计算的最短路径等价于原图最短路径**：
设 `d(u, v)` 是原图中从 `u` 到 `v` 的最短路径距离，`d'(u, v)` 是重赋权后图中从 `u` 到 `v` 的最短路径距离。根据重赋权的定义以及最短路径的性质，对于任意节点对 `u` 和 `v`，有 `d'(u, v) = d(u, v) + h(u) - h(v)`，在计算出重赋权后图中的最短路径 `d'(u, v)` 后，通过还原操作 `d(u, v) = d'(u, v) - h(u) + h(v)` 就可以得到原图中的最短路径距离，所以 Johnson 算法最终计算出的所有节点对之间的最短路径是正确的。
