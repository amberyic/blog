---
title: Attention Is All You Need -- Transformer
date: 2021-09-18
updated: 2021-09-18
categories:
- 机器学习
tags:
- Transformer
- 注意力机制
- Self-Attention
- Multi-Head Attention
keywords: Transformer,Multi-Head Attention,Self-Attention,注意力机制
description: 本文基于论文《Attention Is All You Need》对其中提出的Transformer模型架构进行了拆解，分析了其设计思路和优势。
---

## 概述
Google的翻译团队在《Attention Is All You Need》中提出了他们的Transformer架构，Transformer基于经典的机器翻译Seq2Seq框架，突破性的抛弃了传统的循环和卷积神经网络结构，仅仅依赖注意力机制。在WMT 2014的数据集上取得了很好的成绩。
关于注意力机制，可以翻看我以前的一些文章，对于Attention的原理和变种都有详细的介绍。

**Transformer的三个优势**
- **模型并行度高，使得训练时间大幅度降低。** 循环模型通常是对输入和输出序列的符号位置进行因子计算。通过在计算期间将位置与步骤对齐，它们根据前一步的隐藏状态ht-1和输入产生位置t的隐藏状态序列ht。这种固有的顺序特性阻碍样本训练的并行化，这在更长的序列长度上变得至关重要，因为有限的内存限制样本的批次大小。Transformer架构避免使用循环神经网络并完全依赖于attention机制来绘制输入和输出之间的全局依赖关系，允许进行更多的并行化。
- **可以直接捕获序列中的长距离依赖关系。** 注意力机制允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离。对比LSTM，Attention能够更好的解决长距离依赖问题（Long-Term Dependencies Problem）。
- **自注意力可以产生更具可解释性的模型。** 我们可以从模型中检查注意力分布。各个注意头 (attention head) 可以学会执行不同的任务。

## 模型架构
![Transformer的架构](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109290538512.png)

### Encoder and Decoder Stacks

**编码器**
编码器由N=6个相同的layer组成，layer指的就是上图左侧的单元，最左边有个“Nx”，这里是x6个。每个Layer由两个子层（Sub-Layer）组成,第一个子层是Multi-head Self-attention Mechanism，第二个子层比较简单，是Fully Connected Feed-Forward Network。其中每个子层都加了残差连接(Residual Connection)和层归一化(Layer Normalisation)，因此可以将子层的输出表示为：$\text { LayerNorm }(x+\operatorname{SubLayer}(x))$

**解码器**
解码器同样由N=6个相同layer组成，因为编码器是并行计算一次性将结果直接输出，而解码器是一个词一个词输入，所以解码器除了每个编码器层中的两个子层之外，还插入第三子层，其对编码器堆栈的输出执行multi-head attention。每个子层也都加了残差连接(Residual Connection)和层归一化(Layer Normalisation)。解码器中对self-attention子层进行了修改，以防止引入当前时刻的后续时刻输入，这种屏蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测仅依赖于小于i的位置处的已知输出。

### 注意力
attention函数可以被描述为将query和一组key-value对映射到输出，其中query，key，value和输出都是向量。输出被计算为值的加权求和，其中分配给每个值的权重由query与对应key的兼容性函数计算。这里重点讲解Transformer中用到的几个Attention机制的变种。
![Attention机制的本质思想](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109030902038.png)

#### Scaled Dot-Product Attention
![Scaled Dot-Product Attention](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109290848281.png)
我们将这个Attention称为缩放点积Attention，输入由维度为$d_k$的query和key以及维度为$d_v$的value组成。我们用所有key计算query的点积，然后将每个点积结果除以$\sqrt {d_k}$，并应用softmax函数来获得value的权重。
在实践中，我们同时在一组query上计算attention函数，将它们打包在一起形成矩阵Q，key和value也一起打包成矩阵K和V。我们计算输出矩阵为：
$$\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V$$

Dot-Product Attention和Additive Attention是最常用的两个Attention函数，Dot-Product Attention只是比Scaled Dot-Product Attention少了一个缩放因子，其他都是一样的。Additive Attention使用具有单个隐藏层的前馈网络来计算兼容性函数。虽然两者在理论上的复杂性相似，但在实践中，Dot-Product Attention更快，更节省空间，因为它可以使用高度优化的矩阵乘法来实现。

#### Multi-Head Attention
![Multi-Head Attention](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202109290951436.png)

Multi-Head Attention是利用多个查询，来平行地计算从输入信息中选取多个信息。每个注意力关注输入信息的不同部分，然后再进行拼接。

$$
\begin{aligned}
\text { MultiHead }(Q, K, V) &=\text { Concat }\left(\operatorname{head}_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O} \\
\text { where head }_{\mathrm{i}} &=\text { Attention }\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
$$
> 其中：$W_{i}^{Q} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}$, $W_{i}^{K} \in \mathbb{R}^{d_{\text {model }} \times d_{k}}$, $W_{i}^{V} \in \mathbb{R}^{d_{\text {model }} \times d_{v}}$，$W^{O} \in \mathbb{R}^{h d_{v} \times d_{\text {model }}}$

#### Attention中的mask操作
整个Transformer中包含三种类型的attention,且目的并不相同。
- Encoder的self-attention，考虑到batch的并行化，通常会进行padding，因此会对序列中mask=0的token进行mask后在进行attention score的softmax归一化。
- Decoder中的self-attention，为了避免预测时后续tokens的影所以必须令后续tokens的mask=0，其具体做法为构造一个三角矩阵。
- Decoder中的encode-decoder attention，涉及到decoder中当前token与整个encoder的sequence的计算，所以encoder仍然需要考虑mask。

综上，无论对于哪个类型的attention，在进行sotmax归一化前，都需要考虑mask操作。


### Position-wise Feed-Forward Networks
在编码器和解码器中的每层都包含一个完全连接的前馈网络，该网络分别相同地应用于每个位置，主要是提供非线性变换，之所以是position-wise是因为过线性层时每个位置i的变换参数是一样的。该前馈网络包含两个线性变换，并在第一个的最后使用ReLU激活函数。
$$\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}$$
虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。描述这种情况的另一种方式是两个内核大小为1的卷积。

### Embeddings和Softmax
Embeddings和Softmax跟在常规的序列转换模型中起到的作用是相同的。Embeddings将输入符号和输出符号转换为固定长的向量。线性变换和softmax函数将解码器输出转换为预测的下一个字符的概率。在这个模型中，两个嵌入层和pre-softmax线性变换之间共享相同的权重矩阵。

### Layer Normalization
Layer Normalization是作用于每个时序样本的归一化方法，其作用主要体现在：
- 作用于非线性激活函数前，能够将输入拉离激活函数非饱（防止梯度消失）和非线性区域（保证非线性）；
- 保证样本输入的同分布。

### Positional Encoding
由于我们的模型不包含递归和卷积，为了让模型利用序列的顺序，我们必须注入一些关于标记在序列中的相对或绝对位置的信息。为此，我们将“位置编码”添加到编码器和解码器堆栈底部的输入嵌入中。位置编码具有与词嵌入相同的维度，因此可以将两者相加。
在这项工作中，我们使用不同频率的正弦和余弦函数：
$$\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} &=\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}$$
> 其中，pos是位置，i是维度。

文章中对这块解释的很少，可以参考下面两个链接，详细了解：
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [如何理解Transformer论文中的positional encoding，和三角函数有什么关系？](https://www.zhihu.com/question/347678607)

## 结论
Transformer是第一个完全基于attention的序列转换模型，用multi-headed self-attention取代了encoder-decoder架构中最常用的recurrent layers。

对于翻译任务，Transformer比基于循环或卷积层的体系结构训练更快。 在WMT 2014英语-德语和WMT 2014英语-法语翻译任务中，我们取得了最好的结果。 在前面的任务中，我们最好的模型甚至胜过以前报道过的所有整合模型。

Transformer在长距离的信息捕捉以及计算和性能上的优势明显，后期在GPT、Bert、XLNet等预训练模型上大规模的使用。

## 参考文献
[1][《Transformer: A Novel Neural Network Architecture for Language Understanding》/ Jakob Uszkoreit/ 2017](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
[2][《The Illustrated Transformer》 / Jay Alammar / 2018](https://jalammar.github.io/illustrated-transformer/)
[3][《Attention Is All You Need》/ Ashish Vaswani / 2017](https://arxiv.org/pdf/1706.03762.pdf)
[tensorflow/tensor2tensor / Github](https://github.com/tensorflow/tensor2tensor)
