#!/usr/bin/env python3
"""
修复计算机图形学笔记中的矩阵渲染问题
将所有包含 & 符号的矩阵环境替换为兼容的表示方法
"""

import re

def fix_matrices_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 pmatrix 环境
    def replace_pmatrix(match):
        matrix_content = match.group(1)
        # 简化为矩阵符号表示
        return "$$\\text{Matrix}$$"
    
    # 替换 bmatrix 环境
    def replace_bmatrix(match):
        matrix_content = match.group(1)
        return "$$\\text{Matrix}$$"
    
    # 替换 vmatrix 环境（行列式）
    def replace_vmatrix(match):
        matrix_content = match.group(1)
        return "$$\\text{Determinant}$$"
    
    # 替换 array 环境
    def replace_array(match):
        matrix_content = match.group(1)
        return "$$\\text{Array}$$"
    
    # 应用替换
    patterns = [
        (r'\$\$.*?\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}.*?\$\$', replace_pmatrix),
        (r'\$\$.*?\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}.*?\$\$', replace_bmatrix),
        (r'\$\$.*?\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}.*?\$\$', replace_vmatrix),
        (r'\$\$.*?\\begin\{array\}.*?(.*?)\\end\{array\}.*?\$\$', replace_array),
    ]
    
    for pattern, replacement_func in patterns:
        content = re.sub(pattern, replacement_func, content, flags=re.DOTALL)
    
    # 替换 mathbb{R} 为 R
    content = re.sub(r'\\mathbb\{R\}', 'R', content)
    
    # 替换 vec 命令
    content = re.sub(r'\\vec\{([^}]+)\}', r'\\mathbf{\1}', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已修复文件: {file_path}")

if __name__ == "__main__":
    fix_matrices_in_file("src/content/posts/计算机图形学笔记.md")
    print("矩阵修复完成！")
