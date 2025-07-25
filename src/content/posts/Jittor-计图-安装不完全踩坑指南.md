---
title: Jittor (计图) 安装不完全踩坑指南
published: 2025-06-04 22:46:36
tags: ["Jittor", "计图", "安装", "踩坑", "AI", "深度学习"] # 将 tags 修改为数组，并添加了一些相关的标签作为示例
category: 指南
---

<!--more-->

# Jittor (计图) 安装不完全踩坑指南 📝

此文记录了我在 Windows 11 和 Linux (VMware, WSL)环境下安装 Jittor (计图) 深度学习框架及配置其运行环境时遇到的一些问题和解决过程。

(叠甲，我先前没有任何安装系统经验，也从没用过Linux，犯了很多低级错误)

## 📋 内容概览

- **一、Windows 11 尝试** - Windows 环境下的安装尝试和遇到的问题
- **二、VMware 虚拟机尝试** - 虚拟机环境的限制和问题
- **三、WSL (Windows Subsystem for Linux) 成功** - 最终成功的解决方案
- **四、常见安装问题与解决方案** - CUTLASS 下载失败等常见问题的解决方法
- **五、总结与经验** - 安装过程中的经验总结和建议

==================================================

## 系统配置信息 💻

==================================================

- **操作系统**: Windows 11 Pro 24H2
- **机器类型**: AMD64
- **处理器**: 13th Gen Intel(R) Core(TM) i9-13900H 2.60 GHz
- **总内存**: 31.73 GB
- **GPU 1**: Intel(R) Iris(R) Xe Graphics
- **GPU 2**: NVIDIA GeForce RTX 4060 Laptop GPU
- **NVIDIA 驱动版本**: 572.83
- **CUDA 版本 (宿主机)**: 12.8

---

## 一、Windows 11 尝试 (失败 ❌)

最初尝试在 Windows 11 上通过新建 Anaconda 虚拟环境来安装 Jittor。

1. **Jittor 自动下载依赖**:
    在执行编译步骤前，Jittor 会在缓存文件夹（通常是用户目录下的 `.cache/jittor`）下自动下载并安装其依赖的 CUDA (特定版本) 和 MSVC (编译器) 文件。这个过程下载了大约 4-5GB 的文件，由于网络速度原因，耗时近半小时。

2. **编译错误 1: `lock.py` 不兼容**:
    首次编译时，遇到 `lock.py` 脚本不兼容 Windows 的问题（日志提示引入了 Windows 系统缺失的某个库）。尝试手动修改该文件的 `import` 语句，但未能解决根本问题。

3. **编译错误 2: `data.cc` C2440 错误**:
    随后，Jittor 底层 C++ 代码 `data.cc` 文件在编译时出现 `error C2440: 'type cast'`。错误信息指出，这是由于将类成员函数指针（例如 `jittor::Node::op_info`）直接强制转换为普通函数指针所致。C++ 标准不允许这种跨类型转换，通常需要通过 `std::function` 或静态函数适配器等方式来解决。

    ```cpp
    // 示例性质的代码，非真实源码，仅为说明 C2440 错误类型
    // class MyClass { public: void member_func() {} };
    // typedef void (*FuncPtr)();
    // MyClass obj;
    // FuncPtr ptr = (FuncPtr)&MyClass::member_func; // 这会导致 C2440 错误
    ```

4. **小结**:
    考虑到直接修改 Jittor 核心 C++ 代码的复杂性和潜在风险，且多次重装 Jittor 均未能解决编译问题，判断可能是 Windows 环境与 Jittor 的兼容性或某些深层配置问题。因此，决定放弃 Windows 平台，转向 Linux 环境。

---

## 二、Ubuntu 20.04 (VMware 虚拟机) 尝试 (失败 ❌)

鉴于没有 Linux 使用经验，在朋友的建议和网络教程的帮助下，选择在 VMware Workstation 上安装 Ubuntu 20.04 LTS 虚拟机。

1. **Jittor CPU模式运行成功**:
    在虚拟机中按照 Jittor 官方文档步骤安装，Jittor 本身能够成功安装并利用 CPU 进行运算。

2. **GPU 调用失败**:
    尝试让 Jittor 调用宿主机的 NVIDIA GPU 时，发现无法识别或使用。经过一番排查（包括尝试使用 `.run` 文件安装驱动、查看可用驱动列表等），最终发现关键问题：**VMware Workstation (Player/非授权 Pro 版) 虚拟机通常不支持对消费级 NVIDIA 显卡进行GPU直通或共享 CUDA 计算能力。** 这意味着在我的配置下，虚拟机内的 Ubuntu 无法利用宿主机的 RTX 4060 进行加速。

3. **小结**:
    由于虚拟机无法有效利用 GPU，Jittor 只能在 CPU模式下低效运行，不符合使用需求，因此该方案也宣告失败。

---

## 三、WSL (Windows Subsystem for Linux) 探索

在 VMware 尝试失败后，决定转向 WSL。

### 1. WSL - Ubuntu 24.04 LTS (初次尝试，失败 ❌)

1. **环境搭建**: 在 WSL 中安装了 Ubuntu 24.04 LTS。
2. **对 WSL GPU 支持的误解**:
    配置完 Jittor 后，准备安装 CUDA 及相关驱动。执行 `nvidia-smi` 命令无响应，错误地以为需要在 WSL 内部也安装一套 NVIDIA 驱动。
    > **重要提示**: WSL2可以直接利用宿主机 Windows 上已正确安装的 NVIDIA 驱动和 CUDA 工具包。通常**不需要也禁止**在 WSL 发行版内部再次安装 NVIDIA 内核驱动。
3. **CUDA 安装问题**:
    在尝试安装 CUDA 时，遇到包依赖问题，例如 `dpkg: error: cannot access archive 'libtinfo5_6.3-2_amd64.deb': No such file or directory`。这通常是因为 Ubuntu 24.04 的软件源可能不再包含这个较旧版本的 `libtinfo5` 库，或者需要特定的配置才能找到。
4. **环境混乱**:
    由于在 WSL 内部进行了错误的 NVIDIA 驱动安装尝试，导致 WSL 内的驱动环境变得混乱。
5. **小结**:
    决定删除此 Ubuntu 24.04 WSL 发行版，更换为更稳定且与 Jittor 可能兼容性更好的 Ubuntu 版本。

### 2. WSL - Ubuntu 22.04 LTS (最终成功 ✅)

1. **Python 版本问题**:
    在新的 Ubuntu 22.04 LTS (WSL) 环境中，最初直接使用系统自带或手动安装的 Python 3.9。然而，运行 Jittor 官方安装命令时出现不兼容或报错。
2. **切换 Python 版本**:
    为快速解决，新建了一个 Conda 虚拟环境，并在其中安装了 **Python 3.7**。

    ```bash
    conda create -n jittor_env python=3.7
    conda activate jittor_env
    ```

3. **安装 Jittor**:
    在 Python 3.7 虚拟环境下，按照官方指南安装 Jittor。

    ```bash
    python -m pip install jittor
    ```

4. **g++ 版本问题**:
    Jittor 在首次运行时会进行编译。此时遇到 `g++` 版本过低的问题。
    - **解决方案**: 升级 `g++`。可以通过 `apt` 安装较新版本的 `g++` (例如 `g++-9`, `g++-10` 等，具体版本取决于 Jittor 的需求和 Ubuntu 22.04 的软件源)。如果安装了多个版本，可能需要使用 `update-alternatives` 来设置默认的 `g++` 版本。

        ```bash
        sudo apt update
        sudo apt install g++ # 或者指定版本如 g++-9
        # 如果需要，配置 update-alternatives
        # sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
        # sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
        ```

    - 同时，确保新版 `g++` 的路径被正确添加到了系统环境变量 `PATH` 中，以便 Jittor 的编译脚本能够找到它。

5. **编译和测试**:
    解决 `g++` 版本问题后，Jittor 编译顺利通过。
    执行官方提供的 CUDNN 算子测试命令：

    ```bash
    python3.7 -m jittor.test.test_cudnn_op
    ```

    所有算子均测试正常，显示 Jittor 已能正确利用 GPU (通过 WSL 从宿主机调用) 并启用了 CUDNN 加速。

---

## 四、常见安装问题与解决方案 🔧

### 4.1 CUTLASS 下载失败问题

**问题描述**: 在使用 `jittor.test_install()` 自动配置环境时，可能遇到以下错误：

```text
RuntimeError: MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip
```

**问题原因**:

- CUTLASS 对应的清华网盘链接无法访问
- 程序只创建了一个空的压缩包，导致 MD5 校验失败
- GitHub 上的备用下载路径也可能无法访问

**解决方案**:

#### 方法一：手动替换下载链接

1. **备份原文件**:

   ```bash
   cp /path/to/your/conda/envs/jt/lib/python3.7/site-packages/jittor/compile_extern.py /path/to/backup/jittor_compile_extern_bak.py
   ```

2. **获取新的下载链接和MD5**:
   新的可用下载链接：`https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip`

3. **修改源码文件**:
   编辑 `compile_extern.py` 文件，将原有的下载链接替换为新链接：

   ```bash
   # 找到你的 jittor 安装路径，通常在：
   # /path/to/conda/envs/your_env/lib/python3.7/site-packages/jittor/compile_extern.py

   # 使用 sed 命令替换下载链接
   sed -i "s|https://cloud.tsinghua.edu.cn/f/171e49e5825549548bc4/?dl=1|https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip|g" /path/to/your/jittor/compile_extern.py
   ```

4. **清除缓存并重新测试**:

   ```bash
   rm -rf ~/.cache/jittor/cutlass/
   python -c "import jittor; jittor.test_install()"
   ```

#### 方法二：自动化脚本解决

创建一个自动化脚本来处理这个问题：

```bash
#!/bin/bash

# 自动修复 Jittor CUTLASS 下载问题
# 使用方法：bash fix_jittor_cutlass.sh

echo "🔧 开始修复 Jittor CUTLASS 下载问题..."

# 检测 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ 请先激活你的 conda 环境"
    exit 1
fi

# 查找 jittor 安装路径
JITTOR_PATH=$(python -c "import jittor; print(jittor.__file__)" 2>/dev/null | sed 's/__init__.py//')

if [ -z "$JITTOR_PATH" ]; then
    echo "❌ 未找到 Jittor 安装路径，请确保已安装 Jittor"
    exit 1
fi

COMPILE_EXTERN_FILE="${JITTOR_PATH}compile_extern.py"

echo "📍 找到 Jittor 路径: $JITTOR_PATH"

# 备份原文件
if [ -f "$COMPILE_EXTERN_FILE" ]; then
    cp "$COMPILE_EXTERN_FILE" "${COMPILE_EXTERN_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ 已备份原文件"
else
    echo "❌ 未找到 compile_extern.py 文件"
    exit 1
fi

# 替换下载链接
NEW_URL="https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip"
sed -i "s|https://cloud.tsinghua.edu.cn/f/171e49e5825549548bc4/?dl=1|$NEW_URL|g" "$COMPILE_EXTERN_FILE"

echo "✅ 已更新下载链接为: $NEW_URL"

# 清除缓存
CACHE_DIR="$HOME/.cache/jittor/cutlass/"
if [ -d "$CACHE_DIR" ]; then
    rm -rf "$CACHE_DIR"
    echo "✅ 已清除 CUTLASS 缓存"
fi

echo "🎉 修复完成！现在可以重新运行 jittor.test_install()"
echo "💡 测试命令: python -c \"import jittor; jittor.test_install()\""
```

### 4.2 环境配置建议

为了避免类似问题，建议使用以下命令创建专用环境：

```bash
# 创建包含必要编译工具的环境
conda create -n rtdetr_env -c conda-forge python=3.7 gcc_linux-64 gxx_linux-64 -y
conda activate rtdetr_env

# 安装 Jittor
pip install jittor

# 测试安装
python -c "import jittor; jittor.test_install()"
```

**注意事项**:

- 使用 `conda-forge` 频道确保获取最新的编译工具
- `gcc_linux-64` 和 `gxx_linux-64` 提供了必要的编译环境
- Python 3.7 是经过验证的稳定版本

---

## 五、总结与经验 ✨

- **Windows 兼容性**: 在 Windows 上安装 Jittor 可能会遇到较多编译和环境兼容性问题，直接修改其底层代码风险较高，不建议新手轻易尝试。
- **虚拟机 GPU**: 普通版本的 VMware Workstation (Player 或未授权 Pro) 对消费级 NVIDIA 显卡的 GPU Passthrough 支持有限，难以用于 CUDA 加速。
- **WSL2 是优选**: 对于 Windows 用户，WSL2 是运行 Linux 环境和 Jittor 的更佳选择。
  - **WSL GPU 驱动**: **切记** WSL2 会自动使用宿主机 Windows 的 NVIDIA 驱动。不要在 WSL 内部尝试安装 Linux 版的 NVIDIA 显卡驱动。确保宿主机驱动和 CUDA Toolkit (主要用于提供 `nvcc` 等编译工具和库，Jittor 可能会自带或下载特定版本) 是最新的。
  - **CUDA in WSL**: Jittor 可能会在 WSL 环境中自动下载其适配的 CUDA toolkit，或者你可以遵循 NVIDIA 官方文档在 WSL 中安装 CUDA Toolkit (不含驱动部分)。
- **依赖库版本**: 注意操作系统版本与所需依赖库（如 `libtinfo5`）的兼容性。较新的发行版可能移除了旧的库。
- **编译器版本**: `g++` 等编译器版本对 Jittor 的编译至关重要，需确保版本符合 Jittor 要求。
- **Python 版本**: Jittor 对 Python 版本有一定要求 (如示例中 Python 3.7 成功，3.9 失败)，建议查阅官方文档或使用其推荐的 Python 版本，并始终在**虚拟环境**中进行安装。
- **耐心与排查**: 安装复杂软件时，遇到问题是常态。仔细阅读错误信息，查阅官方文档和社区讨论，逐步排查，是解决问题的关键。

至此，Jittor 总算在 WSL (Ubuntu 22.04 LTS + Python 3.7) 环境下成功安装并配置完毕，可以正常使用 GPU 进行加速了！🎉
