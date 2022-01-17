---
title: 详解Self-Attention和Multi-Head Attention
date: 2021-09-15
updated: 2021-09-15
categories:
- 机器学习
tags:
- 注意力机制
- Self-Attention
- Multi-Head Attention
keywords: 注意力机制,Self-Attention, Multi-Head Attention, 自注意力机制, 多头注意力机制
description: 介绍Self-Attention和Multi-Head Attention，这两个的深入理解是理解transformer的前提。
---

## 概述
Self Attention就是Q、K、V均为同一个输入向量映射而来的Encoder-Decoder Attention，它可以无视词之间的距离直接计算依赖关系，能够学习一个句子的内部结构，实现也较为简单并且可以并行计算。

Multi-Head Attention同时计算多个Attention，并最终得到合并结果，通过计算多次来捕获不同子空间上的相关信息。

## 自注意力机制(Self-Attention)
Self Attention跟Attention机制本质上是一回事，我们在[《Attention机制的基本思想与实现原理》](https://imzhanghao.com/2021/09/01/attention-mechanism/)中已经详细的介绍了Attention机制，这里我们主要讲解一下Self Attention机制的特别之处。

一般我们说Attention的时候，他的输入Source和输出Target内容是不一样的，比如在翻译的场景中，Source是一种语言，Target是另一种语言，Attention机制发生在Target元素Query和Source中所有元素之间。而Self Attention指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制。

Self Attention是在2017年Google机器翻译团队发表的《Attention is All You Need》中被提出来的，它完全抛弃了RNN和CNN等网络结构，而仅仅采用Attention机制来进行机器翻译任务，并且取得了很好的效果，Google最新的机器翻译模型内部大量采用了Self-Attention机制。

### Self-Attention的作用
Self Attention可以捕获同一个句子中单词之间的一些句法特征（比如图展示的有一定距离的短语结构）
![可视化Self Attention机制](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151007413.png)

Self Attention可以捕获同一个句子中单词之间的一些语义特征（比如图展示的its的指代对象Law）。
![可视化Self Attention机制](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151007208.png)

很明显，引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。

但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此外，Self Attention对于增加计算的并行性也有直接帮助作用。这是为何Self Attention逐渐被广泛使用的主要原因。

### Self-Attention的计算过程
![Attention机制的本质思想](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109030902038.png)

**第一步：初始化Q，K，V**
从每个编码器的输入向量（在本例中是每个单词的Embedding向量）创建三个向量。对于每个单词，我们创建一个Query向量、一个Key向量和一个Value向量。这些向量是通过将Embedding乘以我们在训练过程中训练的三个矩阵来创建的。
$$ \begin{array}{l}
Q=W_{q} X \\
K=W_{k} X \\
V=W_{v} X
\end{array}$$

![初始化Q，K，V](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151102923.png)
> 这里Thinking这个单词的Embedding向量是X1，我们用X1乘以WQ的权重矩阵，就可以得到Thinking这个词的Query，即q1。其他的q2、k1、k2等都使用相同的计算方式，这样我们就计算为每个单词都计算了一个Query，一个Key，和一个Value。

> 这些新向量的维度比Embedding向量小。它们的维数是64，而嵌入和编码器输入/输出向量的维数是512。它们不必更小，这是一种架构选择，可以使多头注意力的计算保持不变。

**第二步：计算Self-Attention Score**
假设我们正在计算本例中第一个单词“Thinking”的自注意力。我们需要根据这个词对输入句子的每个词进行评分。当我们在某个位置对单词进行编码时，分数决定了将多少注意力放在输入句子的其他部分上。

得分是通过将查询向量与我们正在评分的各个单词的键向量进行点积来计算的。 因此，如果我们正在处理位置 #1 中单词的自注意力，第一个分数将是q1和k1的点积。第二个分数是q1和k2的点积。

![计算Self-Attention Score](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151123929.png)

**第三步：对Self-Attention Socre进行缩放和归一化,得到Softmax Socre**
对 Step 2 中计算的分数进行缩放，这里通过除以8( 论文中维度是64，这可以让模型有更稳定的梯度，默认值是64，也可以是其它值)，将结果进行softmax归一化。
![计算Softmax Socre](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151127016.png)


**第四步：Softmax Socre乘以Value向量，求和得到Attention Value**
每个Value向量乘以softmax Score得到加权的v1和v2，对加权的v1和v2进行求和得到z1。这样，我们就计算出了第一个词Thinking的注意力值。其他的词用相同的方法进行计算。
![Socre乘以Value向量](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151133617.png)


### Self-Attention计算过程动图
对于Self Attention机制计算过程还有不清楚的地方的同学，推荐看这篇文章[《[Illustrated: Self-Attention》](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a#570c)，里面将计算过程动态的绘制出来，分八个步骤进行讲解。
- Prepare inputs
- Initialise weights
- Derive key, query and value
- Calculate attention scores for Input 1
- Calculate softmax
- Multiply scores with values
- Sum weighted values to get Output 1
- Repeat steps 4–7 for Input 2 & Input 3

![Self-Attention计算过程动图](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151144419.gif)

## 多头注意力机制(Multi-Head Attention)
Multi-Head Attention是利用多个查询，来平行地计算从输入信息中选取多个信息。每个注意力关注输入信息的不同部分，然后再进行拼接。

$$
\begin{aligned}
\text { MultiHead }(Q, K, V) &=\text { Concat }\left(\operatorname{head}_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O} \\
\text { where head }_{\mathrm{i}} &=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
$$
> 其中：$W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}$，$W^{O} \in \mathbb{R}^{h d_{v} \times d_{\text {model }}}$

### Single-Head Attention VS Multi-Head Attention
![Scaled Dot-Product Attention VS Multi-Head Attention](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151148991.png)
上图对比了多头注意力机制的计算过程和多头注意力机制的计算过程。

### Multi-Head Attention的作用
多头注意力的机制进一步细化了注意力层，通过以下两种方式提高了注意力层的性能：
- 扩展了模型专注于不同位置的能力。当多头注意力模型和自注意力机制集合的时候，比如我们翻译“动物没有过马路，因为它太累了”这样的句子的时候，我们想知道“它”指的是哪个词，如果能分析出来代表动物，就很有用。
- 为注意力层提供了多个“表示子空间”。对于多头注意力，我们不仅有一个，而且还有多组Query/Key/Value权重矩阵，这些权重矩阵集合中的每一个都是随机初始化的。然后，在训练之后，每组用于将输入Embedding投影到不同的表示子空间中。多个head学习到的Attention侧重点可能略有不同，这样给了模型更大的容量。
![两个head学到的Attention效果](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109151146086.png)

## 参考文献
[1][Attention机制的基本思想与实现原理](https://imzhanghao.com/2021/09/01/attention-mechanism/)
[2][深度学习中的注意力模型（2017版）/ 张俊林 / 知乎](https://zhuanlan.zhihu.com/p/37601161)
[3][The Illustrated Transformer / Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
[4][Illustrated: Self-Attention / Raimi Karim](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a#570c)
[5][《Attention Is All You Need》/ Ashish Vaswani / 2017](https://arxiv.org/pdf/1706.03762.pdf)
