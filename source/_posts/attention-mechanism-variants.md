---
title: Attention机制变体的介绍与对比
date: 2021-09-06
updated: 2021-09-10
categories:
- 机器学习
tags:
- 注意力机制
- Hard Attention
- Soft Attention
- Global Attention
- Local Attention
- Bahdanau Attention
- Luong Attention
keywords: Hard Attention, Soft Attention, Global Attention, Local Attention, Bahdanau Attention, Luong Attention, attention variants
description: 介绍Attention机制多种变体，对进行对比。包括hard attention和soft attention的对比，global attention 和 local attention的对比，bahdanau attention 和 luong attention的对比。
---

## hard attention vs soft attention

[《Recurrent Models of Visual Attention》](https://arxiv.org/pdf/1406.6247v1.pdf)中Volodymyr Mnih提出了hard attention方法。

[《Neural Machine Translation by Jointly Learning to Align and Translate》](https://arxiv.org/pdf/1409.0473.pdf)中Dzmitry Bahdanau提出了soft attention的方法。

[《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》](https://arxiv.org/pdf/1502.03044.pdf)中Kelvin Xu将这两种方法在Image Caption进行了比较，两种方案生成的注意力效果如下图所示。
![soft attention vs hard attention](https://oss.imzhanghao.com/img/202109060829391.png)

目前我们大量的使用的都是soft attention，虽然hard attention有时能获得更好的训练效果，但是训练难度也会高很多。
这两种attention计算方法主要的差别在于**计算context vector Z的方法不一样**。

### soft attention
* 平均的方法得到Z，对所有区域都关注，只是关注的重要程度不一样。
* 整个模型都是平滑的，可微分的，可以用标准的反向传播算法进行学习。
* 是一种确定（deterministic）的学习过程。

**soft attention计算过程**
x1~Xn分别覆盖图像的一个子部分。为了计算分数 si 来衡量对 xi 的关注程度，我们使用（上下文 C=ht−1）：
$$s_{i}=\tanh \left(W_{c} C+W_{x} X_{i}\right)=\tanh \left(W_{c} h_{t-1}+W_{x} x_{i}\right)$$
我们将 si 传递给 softmax 进行归一化以计算权重 αi。
$$\alpha_{i}=\operatorname{softmax}\left(s_{1}, s_{2}, \ldots, s_{i}, \ldots\right)$$

使用 softmax，αi 加起来为 1，我们用它来计算 x1、x2、x3 和 x4 的加权平均值
$$Z=\sum_{i} \alpha_{i} x_{i}$$
我们把最终得到的Z代替原始输入x，当作LSTM的输入。

![soft attention计算过程](https://oss.imzhanghao.com/img/202109060853898.png)

**soft attention的注意力**
![soft attention的注意力](https://oss.imzhanghao.com/img/202109060831176.png)

### hard attention
* 采样得到Z，权重服从贝努利分布，非0即1，对特定时间特定区域只有关注与不关注。
* 不连续不可导，无法在反向传播中利用梯度更新，使用类似reinforcement learning的方法进行学习。
* 是一种随机（stochastic）的学习过程。

**hard attention计算过程**
x1~xn分别覆盖图像的一个子部分。我们为每个xi计算一个权重αi，并使用它来计算xi作为LSTM输入的加权平均值。αi加起来为1，可以解释为xi是我们应该关注的区域的概率。因此，hard attention不是加权平均值，而是使用αi作为采样率来选择一个xi作为 LSTM 的输入。

$$Z \sim x_{i}, \alpha_{i}$$

![hard attention计算过程](https://oss.imzhanghao.com/img/202109060854596.png)

**hard attention的注意力**
![hard attention的注意力](https://oss.imzhanghao.com/img/202109060831750.png)

## global attention vs local attention
global attention和local attenion的区别在于“注意力”是放在所有源位置还是仅放在几个源位置。
在[《Effective Approaches to Attention-based Neural Machine Translation 》](https://arxiv.org/pdf/1508.04025.pdf)中，Luong做了详细的说明和对比。


### global attention
![global attention](https://oss.imzhanghao.com/img/202109070602175.png)
全局注意力模型的思想是在推导上下文向量ct的时候考虑编码器的所有隐藏状态,在该模型类型中，通过将当前目标隐藏状态ht与每个源隐藏状态hs进行比较，得出大小等于源侧时间步数的可变长度对齐向量。
$$
\begin{aligned}
\boldsymbol{a}_{t}(s) &=\operatorname{align}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right) \\
&=\frac{\exp \left(\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right)\right)}{\sum_{s^{\prime}} \exp \left(\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s^{\prime}}\right)\right)}
\end{aligned}
$$

这里的score函数有下面三种选择：内积、general和concat，结果表明general效果比较好。
$$
\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right)=\left\{\begin{array}{ll}
\boldsymbol{h}_{t}^{\top} \overline{\boldsymbol{h}}_{s} & \text { dot } \\
\boldsymbol{h}_{t}^{\top} \boldsymbol{W}_{\boldsymbol{a}} \overline{\boldsymbol{h}}_{s} & \text { general } \\
\boldsymbol{v}_{a}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{a}}\left[\boldsymbol{h}_{t} ; \overline{\boldsymbol{h}}_{s}\right]\right) & \text { concat }
\end{array}\right.
$$

### local attention
![local attention](https://oss.imzhanghao.com/img/202109070603083.png)
为了进一步减少计算代价，在解码过程的每一个时间步仅关注输入序列的一个子集，于是在计算每个位置的attetnion是会固定一个上下文窗口，而不是在全局范围计算attention。局部注意力只会关注部分隐状态，首先对于第t个位置的输出词语，我们在原文中找到它的一个对应位置pt。然后我们在对齐位置pt前后扩张D个长度，得到一个范围[pt-D,pt+D],这个范围就是现在Ct所能够接触到的所有可以参与attention计算的隐藏层范围，最后在这个范围内计算局部对齐权重即可。

从前面的描述我们可以知道，该机制的重点就在于如何确定预测词对应的隐状态，即找到一个合适的pt，论文中提出了两种方法：
**monotonic alignment(local-m)**
简单设置$p_{t}=t$，即假设源序列和目标序列大致单调对齐，D随经验选取。这种单一映射的方法显然太粗暴。

**predictive alignment(local-p)**
不认为原序列和目标序列大致单调对齐，预测一个对齐位置。
$$p_{t}=S \cdot \operatorname{sigmoid}\left(\boldsymbol{v}_{p}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{p}} \boldsymbol{h}_{t}\right)\right)$$
其中$v_{p}^{T}$和$W_{p}$都是可学习的参数，S是source的长度，作为sigmoid的结果，pt∈[0, S]。为了提高pt附近的对齐的得分，以pt为中心放置一个高斯分布。我们的对齐权重现在定义为:
$$\boldsymbol{a}_{t}(s)=\operatorname{align}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right) \exp \left(-\frac{\left(s-p_{t}\right)^{2}}{2 \sigma^{2}}\right)$$
经验上标准差设置为$\sigma=D / 2$，pt是一个真实的数字，s是一个以pt为中心的窗口中的整数。

### 总结
目前我们大量使用的都是global attention，因为local attetnion在encoder不长时，计算量并没有减少，并且位置向量pt的预测并不是非常准确，直接影响到local attention的效果。
![Alignment functions](https://oss.imzhanghao.com/img/202109070950753.png)

## bahdanau attention vs luong attention
luong attention和bahdanau attention是比较流行和经典的两种attention机制实现，是用作者名字命名的，分别是在Minh-Thang Luong的[《Effective Approaches to Attention-based Neural Machine Translation》](https://arxiv.org/pdf/1508.04025.pdf)和Dzmitry Bahdanau的[《Neural Machine Translation by Jointly Learning to Align and Translate》](https://arxiv.org/pdf/1409.0473.pdf)中被提出来的方法。

![bahdanau attention vs luong attention](https://oss.imzhanghao.com/img/202109070948179.png)
这两种机制很相似，区别Luong在他的paper的3.1章节中进行了说明：
1.在Bahdanau Attention机制中，第t步的注意力对齐中，使用的是Decoder中第t-1步的隐藏状态$h_{t-1}$和Encoder中所有的隐藏状态$\overline{\mathbf{h}}_{s}$加权得出的，但是在Luong使用的是第t步的隐藏状态$h_{t}$。
2.在Bahdanau Attention机制中，decoder在第t步时，输入是由$c_t$和Decoder第t-1步的隐藏状态$h_{t-1}$拼接得出的，得到第t步的隐藏状态$h_{t}$并直接输出$\hat{\mathbf{y}}_{t+1}$。而 Luong Attention 机制在 decoder部分建立了一层额外的网络结构，输入是有$c_t$和Decoder第t步的隐藏状态$h_{t}$拼接作为输入，得到第t步的隐藏状态$\tilde{\mathbf{h}}_{t}$并输出$\hat{\mathbf{y}}_{t}$。
3.Bahdanau Attention 机制只尝试了concat作为对齐函数，而Luong Attention 机制的论文在多种对齐函数上做了实验。

## 参考资料
[1][Soft & hard attention / Jonathan Hui](https://jhui.github.io/2017/03/15/Soft-and-hard-attention/)
[2][Soft attention vs. hard attention](https://stackoverflow.com/questions/35549588/soft-attention-vs-hard-attention)
[3][Show, Attend and Tell: Neural Image Caption Generation with Visual Attention / kelvin Xu](https://arxiv.org/pdf/1502.03044.pdf)
[4][Effective Approaches to Attention-based Neural Machine Translation / Minh-Thang Luong](https://arxiv.org/pdf/1508.04025.pdf)
[5][Neural Machine Translation by Jointly Learning to Align and Translate / Dzmitry Bahdanau](https://arxiv.org/pdf/1409.0473.pdf)
[6][一文看懂 Bahdanau 和 Luong 两种 Attention 机制的区别](https://zhuanlan.zhihu.com/p/129316415)
[7][Attention Variants/ Liang Jingxi](http://cnyah.com/2017/08/01/attention-variants/)
