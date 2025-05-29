# 项目说明
## 用途
该项目用于LBM_DDF模拟RB热对流系统，可修改边界条件等内容将其拓展到其他计算流体力学算例。
该项目目前仅发布单GPU加速版本，和单节点多GPU并行加速版本。

若需要多节点超算集群并行方案，或有任何问题，欢迎联系: gaoyuqing536@163.com

## 项目结构
主要源文件(src)：
- main.cu：主程序入口
- collision.cu：碰撞步骤实现
- streaming.cu：流动步骤实现
- globals.cu：全局变量定义
- initial.cu：初始化相关代码
- macrovar.cu：宏观变量计算
- output.cu：输出相关功能
- parameters.h：参数配置
- rb3d.h：主要头文件

并行方案(Parallel):
- main.cu：主程序入口
- collision.cu：碰撞步骤实现
- streaming.cu：流动步骤实现
- globals.cu：全局变量定义
- initial.cu：初始化相关代码
- macrovar.cu：宏观变量计算
- output.cu：输出相关功能
- multi_gpu.cu：多GPU并行管理
- multi_gpu.h：多GPU并行头文件
- rb3d.h：主要头文件
- parameters.h：参数配置

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