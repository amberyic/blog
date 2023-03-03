---
title: Attention机制的基本思想与实现原理
date: 2021-09-01
updated: 2021-09-10
categories:
- 机器学习
tags:
- 注意力机制
- 编码器-解码器
keywords: 注意力机制,编码器-解码器,Attention Value,
description: 从人类注意力机制，到编码器-解码器框架的缺陷，引入注意力机制的必要性。详细介绍了Attention的基本思想，Attention Value的计算方法。
---

## 概述
Attention（注意力）机制如果浅层的理解，跟他的名字非常匹配。他的核心逻辑就是**从关注全部到关注重点**。

### 研究进展
Attention机制最早在视觉领域提出，2014年Google Mind发表了《Recurrent Models of Visual Attention》，使Attention机制流行起来，这篇论文采用了RNN模型，并加入了Attention机制来进行图像的分类。

2015年，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中，将attention机制首次应用在nlp领域，其采用Seq2Seq+Attention模型来进行机器翻译，并且得到了效果的提升。

2017年，Google机器翻译团队发表的《Attention is All You Need》中，完全抛弃了RNN和CNN等网络结构，而仅仅采用Attention机制来进行机器翻译任务，并且取得了很好的效果，注意力机制也成为了大家的研究热点。

### 人类的视觉注意力
Attention 机制很像人类看图片的逻辑，当我们看一张图片的时候，我们并没有看清图片的全部内容，而是将注意力集中在了图片的焦点上。下图形象的展示了人类在看到一副图像时是如何高效分配有限的注意力资源的，其中红色区域表明视觉系统更关注的目标。很明显对于如图所示的场景，人们会把注意力更多的投入到人的脸部，文本的标题以及文章首句等位置。

![人类的视觉注意力](https://oss.imzhanghao.com/img/20210526141037.png)

## Encoder-Decoder的缺陷
上一篇文章我们已经介绍了[Encoder-Decoder模型框架](https://imzhanghao.com/2021/08/26/encoder-decoder/)，不了解的朋友可以返回去再看一下。

![Encoder-Decoder框架](https://oss.imzhanghao.com/img/20210526143504.png)

生成目标句子单词的过程成了下面的形式：
$$\begin{array}{l}
\mathbf{Y}_{\mathbf{1}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}\right) \\
\mathbf{Y}_{2}=\mathbf{f} \mathbf{1}\left(\mathbf{C}, \mathbf{Y}_{\mathbf{1}}\right) \\
\mathbf{Y}_{3}=\mathbf{f} \mathbf{1}\left(\mathbf{C}, \mathbf{Y}_{\mathbf{1}}, \mathbf{Y}_{2}\right)
\end{array}$$
其中f1是Decoder的非线性变换函数。从这里可以看出，在生成目标句子的单词时，不论生成哪个单词，它们使用的输入句子Source的语义编码C都是一样的，没有任何区别。

关于 Encoder-Decoder，有2点需要强调：
- 不论输入和输出的长度是多少，中间的"语义编码C"长度都是固定的。
- 根据不同的任务可以选择不同的编码器和解码器（可以是一个RNN，但通常是其变种LSTM或者GRU）

语义编码C是由句子Source的每个单词经过Encoder编码产生的，这意味着不论是生成哪个单词，其实句子Source中任意单词对生成某个目标单词来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

我们拿机器翻译来解释一下注意力在Encoder-Decoder模型中的作用就更好理解了，比如输入的是英文句子：Tom chase Jerry，Encoder-Decoder框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。

在翻译“杰瑞”这个中文单词的时候，没有注意力的模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，显然“Jerry”对于翻译成“杰瑞”更重要，但是没有注意力的模型是无法体现这一点的。

没有引入注意力的模型在输入句子比较短的时候问题不大，但是如果输入句子比较长，此时所有语义完全通过一个中间语义向量来表示，单词自身的信息已经消失，可想而知会丢失很多细节信息，这也是为何要引入注意力模型的重要原因。

## Attention机制
如果引入Attention模型的话，应该在翻译“杰瑞”的时候，体现出英文单词对于翻译当前中文单词不同的影响程度，比如给出类似下面一个概率分布值：
$$(Tom,0.3) (Chase,0.2) (Jerry,0.5)$$

每个英文单词的概率代表了翻译当前单词“杰瑞”时，注意力分配模型分配给不同英文单词的注意力大小。这对于正确翻译目标语单词肯定是有帮助的，因为引入了新的信息。

### 语义编码的计算方法
目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的Ci。理解Attention模型的关键就是这里，即由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的Ci。增加了注意力模型的Encoder-Decoder框架理解起来如图所示。
![引入注意力模型的Encoder-Decoder框架](https://oss.imzhanghao.com/img/20210526150157.png)
即生成目标句子单词的过程成了下面的形式：
$$\begin{array}{l}
\mathbf{Y}_{\mathbf{1}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{\mathbf{1}}\right) \\
\mathbf{Y}_{2}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{2}, \mathbf{Y}_{\mathbf{1}}\right) \\
\mathbf{Y}_{3}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{3}, \mathbf{Y}_{\mathbf{1}}, \mathbf{Y}_{2}\right)
\end{array}$$

而每个Ci可能对应着不同的源语句子单词的注意力分配概率分布，比如对于上面的英汉翻译来说，其对应的信息可能如下：
![翻译各个单词对应的信息](https://oss.imzhanghao.com/img/20210526150927.png)
其中，f2函数代表Encoder对输入英文单词的某种变换函数，比如如果Encoder是用的RNN模型的话，这个f2函数的结果往往是某个时刻输入xi后隐层节点的状态值；g代表Encoder根据单词的中间表示合成整个句子中间语义表示的变换函数，一般的做法中，g函数就是对构成元素加权求和，即下列公式：
$$C_{i}=\sum_{j=1}^{L_{x}} a_{i j} h_{j}$$

假设Ci中那个i就是上面的“汤姆”，那么Tx就是3，代表输入句子的长度，h1=f(“Tom”)，h2=f(“Chase”),h3=f(“Jerry”)，对应的注意力模型权值分别是0.6,0.2,0.2，所以g函数就是个加权求和函数。如果形象表示的话，翻译中文单词“汤姆”的时候，数学公式对应的中间语义表示Ci的形成过程类似下图：
![Ci的形成过程](https://oss.imzhanghao.com/img/20210526152842.png)

### 注意力分配的方法
上面的注意力(a11、a12、a13)是我们人工分配的，那模型中注意力是怎么计算的呢？
这就需要用到对齐模型，来衡量encoder端的位置j的词，对于decoder端的位置i个词的对齐程度（影响程度），换句话说：decoder端生成位置i的词时，有多少程度受encoder端的位置j的词影响。对齐模型的计算方式有很多种，不同的计算方式，代表不同的Attention模型，最简单且最常用的的对齐模型是dot product乘积矩阵，即把target端的输出隐状态ht与source端的输出隐状态进行矩阵乘。下面是常见的对齐计算方式：
![常见的对齐计算方法](https://oss.imzhanghao.com/img/20210526154026.png)
其中,Score(ht,hs) = aij表示源端与目标单单词对齐程度。常见的对齐关系计算方式有：点乘（Dot product），权值网络映射（General）和concat映射几种方式。

**注意力系数的计算过程**
![注意力的分配过程](https://oss.imzhanghao.com/img/202109030614911.png)
对于采用RNN的Decoder来说，在时刻i，如果要生成yi单词，我们是可以知道Target在生成Yi之前的时刻i-1时，隐层节点i-1时刻的输出值Hi-1的，而我们的目的是要计算生成Yi时输入句子中的单词“Tom”、“Chase”、“Jerry”对Yi来说的注意力分配概率分布，那么可以用Target输出句子i-1时刻的隐层节点状态Hi-1去一一和输入句子Source中每个单词对应的RNN隐层节点状态hj进行对比，即通过函数F(hj,Hi-1)来获得目标单词yi和每个输入单词对应的对齐可能性，这个F函数在不同论文里可能会采取不同的方法，然后函数F的输出经过Softmax进行归一化就得到了符合概率分布取值区间的注意力分配概率分布数值。

**Attention计算过程动图**
对于Attention机制计算过程还有不清楚的地方的同学，推荐看这篇文章[《Attn: Illustrated Attention》](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3#0458)，里面将计算过程动态的绘制出来，分六个步骤进行讲解。
![Attention计算过程动图](https://oss.imzhanghao.com/img/202109030859123.gif)

## Attention原理
上面我们都是在Encoder-Decoder的框架下讨论注意力机制，但是注意力机制本身是一种通用的思想，并不依赖于特定框架。
现在我们抛开Encoder-Decoder来讨论下Attention的原理。

Attention机制其实就是一系列注意力分配系数，也就是一系列权重参数罢了。

### 主流Attention框架
Attention是一组注意力分配系数，那么它是怎样实现的？这里要提出一个函数叫做attention函数，它是用来得到Attention value的。比较主流的attention框架如下：
![Attention机制的本质思想](https://oss.imzhanghao.com/img/202109030902038.png)
我们将Source中的元素想像成一系列的<Key,Value>数据对，此时指定Target中的某个元素Query，通过计算Query和各个元素相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，得到最终的Attention值。

本质上Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。

### 另一个角度理解
可以将Attention机制看做**软寻址**，序列中每一个元素都由key(地址)和value(元素)数据对存储在存储器里，当有query=key的查询时，需要取出元素的value值(也即query查询的attention值)，与传统的寻址不一样，它不是按照地址取出值的，它是通过计算key与query的相似度来完成寻址。这就是所谓的软寻址，它可能会把所有地址(key)的值(value)取出来，上步计算出的相似度决定了取出来值的重要程度，然后按重要程度合并value值得到attention值，此处的合并指的是加权求和。

### 三阶段计算Attention过程
基于上面的推广，我们可以用如下方法描述Attention计算的过程。
Attention函数共有三步完成得到Attention值。
- 阶段1:Query与Key进行相似度计算得到权值
- 阶段2:对上一阶段的计算的权重进行归一化
- 阶段3:用归一化的权重与Value加权求和，得到Attention值

![Attention机制三阶段计算Attention过程](https://oss.imzhanghao.com/img/202109030903758.png)


## 参考文献
[1][深度学习中的注意力模型（2017版）/ 张俊林 / 知乎](https://zhuanlan.zhihu.com/p/37601161)
[2][Attention_Network_With_Keras / Choco31415 / github](https://github.com/Choco31415/Attention_Network_With_Keras)
[3][Attn: Illustrated Attention / Raimi Karim](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
