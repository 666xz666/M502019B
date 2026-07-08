import torch


def tensor_sub():
    m = torch.randint(high=10, low=0, size=(1, 3))
    n = torch.randint(high=10, low=0, size=(2, 1))
    print("m: ", m)
    print("n: ", n)
    print("m - n: ")
    print(m - n)
    print("torch.sub(m, n): ")
    print(torch.sub(m, n))
    print("m.sub_(n), m:")
    m.sub_(n)
    print(m)


"""
m:  tensor([[9, 2, 6]])
n:  tensor([[9],
        [2]])
m - n: 
tensor([[ 0, -7, -3],
        [ 7,  0,  4]])
torch.sub(m, n): 
tensor([[ 0, -7, -3],
        [ 7,  0,  4]])
m.sub_(n), m:
Traceback (most recent call last):
  File "/home/experiments/M502019B/1/basic.py", line 28, in <module>
    main()
    ~~~~^^
  File "/home/experiments/M502019B/1/basic.py", line 24, in main
    tensor_sub()
    ~~~~~~~~~~^^
  File "/home/experiments/M502019B/1/basic.py", line 14, in tensor_sub
    m.sub_(n)
    ~~~~~~^^^
RuntimeError: output with shape [1, 3] doesn't match the broadcast shape [2, 3]

首先定义两个张量，矩阵 M 维度是 1 行 3 列，矩阵 N 维度是 2 行 1 列，下面分三种减法方式逐一说明计算过程、结果与差异。
第一种写法直接使用 m - n 进行减法运算
原始 m 内容是 [[9, 2, 6]]，形状 (1,3)，原始 n 内容是 [[9],[2]]，形状 (2,1)。
PyTorch 会触发广播机制，从最右侧维度开始对齐扩充维度：m 的最后一维是 3，n 最后一维是 1，n 会横向复制 3 列变成两行三列；m 倒数第二维是 1，n 倒数第二维是 2，m 会纵向复制两行变成两行三列。
扩充之后 m 变为 [[9,2,6],[9,2,6]]，n 变为 [[9,9,9],[2,2,2]]，对应位置逐个数字相减，得到结果 [[0, -7, -3],[7, 0, 4]]，运算会生成全新张量，不会改动原本 m 和 n 的数据，不会报错。
第二种写法调用 torch.sub (m, n) 完成减法
底层运算逻辑和直接写 m 减 n 完全一致，同样遵循广播规则扩充两个张量到 2 行 3 列后逐元素相减，最终输出一模一样的结果张量，同样创建新结果张量，不修改原始输入张量，区别仅在于这是官方函数调用形式，还可以额外传入参数指定输出存放位置、设置数值缩放等拓展功能，不会出现报错。
第三种写法使用 m.sub_(n) 带下划线的原地减法操作
前两步广播推演后最终计算结果需要是 2 行 3 列的张量，但带下划线代表原地运算，意思是直接在原张量 m 的内存空间里修改数值，绝对不能更改 m 原本的形状。
m 最开始固定是 1 行 3 列，而广播计算要求输出尺寸为 2 行 3 列，原张量空间无法扩容适配新维度，维度无法匹配，因此程序直接抛出运行时错误。
三者核心区别总结：
前两种方式属于非原地计算，自动执行广播机制，生成新张量保存结果，原始张量不会被修改；第三种是原地覆盖修改，不允许借助广播扩大张量维度，只能在原有张量固定尺寸内修改元素，一旦广播后目标尺寸和原张量尺寸不一致就会直接报错。
整个计算过程核心就是广播机制对两个维度不匹配张量做维度补齐，行向量向下复制行、列向量向右复制列，补齐成相同大小矩阵后再逐位做减法。
"""


def mm():
    p = torch.randn((3, 2)) * 0.01
    q = torch.randn((4, 2)) * 0.01
    print("p: ", p)
    print("q: ", q)
    q_t = q.transpose(-2, -1)
    print("q_t: ", q_t)
    print("p @ q_t: ", p @ q_t)


"""
p:  tensor([[ 0.0067, -0.0045],
        [ 0.0009, -0.0132],
        [ 0.0021, -0.0027]])
q:  tensor([[-0.0014,  0.0021],
        [ 0.0114, -0.0094],
        [-0.0019,  0.0178],
        [-0.0012, -0.0158]])
q_t:  tensor([[-0.0014,  0.0114, -0.0019, -0.0012],
        [ 0.0021, -0.0094,  0.0178, -0.0158]])
p @ q_t:  tensor([[-1.8919e-05,  1.1878e-04, -9.2624e-05,  6.2657e-05],
        [-2.8867e-05,  1.3481e-04, -2.3677e-04,  2.0857e-04],
        [-8.5028e-06,  4.8620e-05, -5.1422e-05,  3.9752e-05]])
"""


def test_grad():
    x = torch.tensor(1.0, requires_grad=True)

    y1 = x**2

    with torch.no_grad():
        y2 = x**3

    y3 = y1 + y2

    y3.backward()

    print("x.grad: ", x.grad)

"""
with torch.no_grad() 上下文内的计算不会构建计算图, y2不会做反向传播
x.grad:  tensor(2.)
"""


def main():
    # tensor_sub()
    # mm()
    test_grad()


if __name__ == "__main__":
    main()
