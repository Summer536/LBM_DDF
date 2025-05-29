# 项目说明
## 用途
该项目用于LBM_DDF模拟RB热对流系统，可修改边界条件等内容将其拓展到其他计算流体力学算例。
该项目目前仅发布单GPU加速版本，和单节点多GPU并行加速版本，若需要多节点超算集群并行方案请联系: gaoyuqing536@163.com

## 代码优化

### 循环展开优化
在collision.cu文件中添加了`#pragma unroll`指令来实现循环展开优化。这种优化通过减少循环控制开销来提高程序性能，同时保持计算结果的准确性。

### 编译优化
在Makefile中添加了以下编译选项：
- `-O3`：启用最高级别的优化
- `--use_fast_math`：启用快速数学运算优化

对于循环展开优化，我们使用CUDA的`#pragma unroll`指令来实现，这比编译选项`--unroll-loops`更可靠和可控。这些优化旨在提高CUDA程序的执行效率，特别是在处理大规模数值计算时。

## 项目结构
主要源文件：
- main.cu：主程序入口
- collision.cu：碰撞步骤实现
- streaming.cu：流动步骤实现
- globals.cu：全局变量定义
- initial.cu：初始化相关代码
- macrovar.cu：宏观变量计算
- output.cu：输出相关功能
- parameters.h：参数配置
- rb3d.h：主要头文件

## 编译和运行
使用以下命令编译项目：
```bash
make
```

运行程序：
```bash
./rb3d
```

清理编译文件：
```bash
make clean
```