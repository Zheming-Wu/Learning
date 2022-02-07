# 复合材料力学python库说明
## 0. 简介
根据矫桂琼、贾普荣所著《复合材料力学》一书所编写的用于复合材料单层及层合板性能计算的库。包含两个类：1.layer类，用于定义并计算复合材料单层的各性能参数；2.compound类，用于定义并计算复合材料层合板的各项性能参数。

## 1. layer类
### 1.1 复合材料单层的定义：

针对复合材料单层设计的类，内置两个基本的材料：HT3/5224，和HT3/QY8911，可以通过如下函数进行layer类的初始化：

```python
import composite as cp
A = cp.layer('HT3/5224')
```

也可以通过手动输入材料性能参数的方式进行定义：

- 通过输入弹性常数$E_L,E_T,v_{LT},G_{LT}$初始化：

```python
import composite as cp
A = cp.layer()
A.set_by_E(135,8.8,0.33,4.47)
```

- 通过输入刚度参数$Q_{11},Q_{22},Q_{12},Q_{66}$初始化：

```python
import composite as cp
A = cp.layer()
A.set_by_Q(136,8.86,2.92,4.47)
```

- 通过输入柔度参数$S_{11},S_{22},S_{12},S_{66}$初始化：

```python
import composite as cp
A = cp.layer()
A.set_by_S(7.41e-3,114e-3,-2.44e-3,224e-3)
```

