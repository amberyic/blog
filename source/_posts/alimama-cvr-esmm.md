---
title: 阿里CVR预估模型ESMM
date: 2020-11-06
categories:
- 机器学习
tags:
- ESMM
- Multi-Task Learning
- MTL
- CVR
keywords: ESMM,Multi-Task Learning,MTL,CVR,多任务学习,多目标学习,阿里巴巴,点击率预估,转化率预估,Entire Space Multi-Task Model
description: ESMM是阿里巴巴践行多目标任务优化模型的一个实践成果，算法紧密结合阿里的电商场景，是一个非常巧妙的模式。本文通过介绍CVR预估过程中存在的问题和挑战，讲解了阿里ESMM算法的模型结构、损失函数以及创新算法。
---

## CVR预估的场景
在诸如信息检索、推荐系统、在线广告投放系统等工业级的应用中准确预估转化率（post-click conversion rate，CVR）是至关重要的。
- 对于DSP(Demand-Side Platform,需求方平台)中CPI(Cost Per Install，按每次安装付费)广告的投放，广告主只为他们推广应用每次的下载付费，所以对于DSP平台来说，让用户仅仅点击广告是不够的，还需要用户点击广告后去应用市场下载应用才能获得收益。DSP平台的**利润=曝光量×点击率×转化率×转化收益-曝光量×单位曝光成本**，这就导致DSP平台不仅仅需要预估用户对广告的点击率，还需要预估用户点击广告后的转化率。
- 对于SSP(Supply Side Platform, 供应方平台)中oCPC(Optimized Cost Per Click, 以目标转化为优化方式的点击出价)出价的场景中，需要使用pCVR调整每次点击的出价，从而实现平台和广告客户的双赢。
- 在推荐系统中，是希望能够借助推荐系统提升整站的GMV，**GMV=流量×点击率×转化率×客单价**,可见点击率和转化率是优化目标非常重要的两个因子，而这两个指标的共同优化，其实就是一个[多目标排序问题](https://imzhanghao.com/2020/10/25/multi-task-learning/).

## CVR预估的挑战
**样本选择偏差（Sample Selection Bias，SSB）**
![样本选择偏差](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/样本选择偏差.png)
传统的CVR训练用的是点击数据，用户点击后未转化为负样本，点击后转化为正样本。但点击事件仅仅是整个曝光空间的一个子集，数据分布是不一致的，模型的泛化能力就会受到影响。

**数据稀疏问题（data sparsity Problem， DS）**
在实践中，点击率一般是比较低的，曝光数量远远大于点击数量，所以训练CVR模型的数据通常比CTR任务的少的多，训练数据的稀疏性使得CVR模型的拟合变得相当困难。

**延迟反馈（delayed feedback）**
用户发生点击后，可能需要较长时间才能发生转化，负样本可能是假性负样本，这给建模带来了很多困难。

## 目前已有的解决方案
**样本选择偏差**
- 缺失值作为负样本（All Missing As Negative，AMAN）采用随机抽样策略把选择未点击的展示作为负示例，这在某种程度上可以减轻样本选择偏差的问题，但通常会会导致预测值偏低。

- 无偏采样（Unbias Sampling）通过蒙特·卡罗拒绝采样法（Rejection Sampling）来拟合观测值的真实基础分布，从而解决了CTR建模中的样本选择偏差问题。但是，通过拒绝概率的除法对样本加权时，可能会遇到数值不稳定性。

**数据稀疏问题**
- 建立了基于不同特征的分层估计器，并与逻辑回归模型相结合，但是，它依靠先验知识来构造层次结构，这很难在具有数千万用户和项目的推荐系统中应用。
- 过采样方法，复制了罕见分类的样本，这有助于减轻数据的稀疏性，但对采样率敏感。

**延迟反馈**
这个问题的解决方案，推荐阅读:[Modeling delayed feedback in display advertising by Olivier Chapelle. KDD 2014](http://wnzhang.net/share/rtb-papers/delayed-feedback.pdf)。主要思想就是对于还未观察到conversion的样本，不直接将其当做负样本，而是当前考虑click已发生的时间长短给模型不同大小的gradient.
> 这不是本算法解决的重点问题，因为阿里推荐系统中转化反馈的延迟是可以接受的。

总之，在CVR建模的情况下，SSB和DS问题都没有得到很好的解决，并且上述方法都没有利用动作的顺序信息。

## Entire Space Multi-Task Model
为了解决上述问题，阿里算法团队提出了关于CVR预估的新模型ESSM，[《Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate》](https://arxiv.org/pdf/1804.07931.pdf)，发表在SIGIR'2018。

### 定义
我们以电子商务网站推荐系统中的CVR建模为例，给定推荐的商品，用户可以点击感兴趣的商品，然后再购买其中一些。换句话说，用户操作遵循曝光（impression）$\rightarrow$ 点击（click）$\rightarrow$ 转换（conversion）的顺序模式。以此方式，**CVR建模是指预估点击后转化率的任务**，即$pCVR = p(conversion | click, impression)$。

假设我们观察的数据空间是$\mathcal{S}=\left.\left\{\left(\boldsymbol{x}_{i}, y_{i} \rightarrow z_{i}\right)\right\}\right|_{i=1} ^{N}$，其中*N*代表总曝光数。
- $\boldsymbol{x}$代表曝光时能够观察到的特征向量，通常是具有多个字段的高维稀疏向量，例如用户相关的特征，商品相关的特征等，属于特征空间。
- $y$和$z$是二进制标签，$y=1$或$z=1$分别表示点击或转化事件发生，属于标签空间。$y→z$揭示了点击和转化标签的顺序相关性，即在发生转化事件时总会先有一个点击。
- pCTR是点击率，$pCTR=p(z=1 \mid \boldsymbol{x})$
- pCVR是转化率，$pCVR=p(z=1 \mid y=1, \boldsymbol{x})$
- pCTCVR是点击转化率，即既点击又转化的概率，$pCTCVR=p(y=1, z=1 \mid \boldsymbol{x})$

点击转化率（pCTCVR），点击率（pCTR）与转化率（pCVR）关系如下：
$$\underbrace{p(y=1, z=1 \mid \boldsymbol{x})}_{pCTCVR}=\underbrace{p(y=1 \mid \boldsymbol{x})}_{pCTR} \times \underbrace{p(z=1 \mid y=1, \boldsymbol{x})}_{pCVR}$$

根据上式，可以看到三者的关系非常明确，那么也就意味着，我只要得到了三者中的二者就可以方便地估计剩下的一个参数了。

### 预估CVR
常规的CVR模型只用了点击以后的样本去预估pCVR，这个会有样本选择偏差的问题，好消息是pCTCVR和pCTR是可以在全量数据集上学习的，我们变换一下上面的公式，就可以根据pCTCVR和pCTR这两个在全量数据集上学习到的值计算出pCVR。
$$p(z=1 \mid y=1, \boldsymbol{x})=\frac{p(y=1, z=1 \mid \boldsymbol{x})}{p(y=1 \mid \boldsymbol{x})}$$
然而，实际上，pCTR很小，除以pCTR会引起数值不稳定，而且有可能是的pCVR超过1，这明显不合理。 

ESMM通过乘法形式避免了这种情况，就是用全样本使用一个模型来同时学习pCTR以及pCVR，然后二者相乘拟合pCTCVR，pCTR预估以及pCTCVR预估是可以使用全样本训练的。

### 网络结构
![ESMM网络结构](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/ESMM网络结构.png)
ESMM是一个双塔模型，模型采用CTCVR和CTR来学习CVR，模型结构如上图。主要由两个子网组成：左侧所示的CVR网络和右侧所示的CTR网络，它们是共享底层Embedding的，只是上面的不一样，一个用来预测CVR，这个可以在全样本空间上进行训练，另一个是用来预测CTR，CTR是一个辅助任务。最后的pCTCVR可以在全样本空间中训练。

### 损失函数
ESMM的损失函数定义为以下公式。它由CTR和CTCVR任务的两个损失项组成，这些损失项是根据所有展示的样本计算得出的，而没有使用CVR任务的损失。
$$L\left(\theta_{cvr}, \theta_{ctr}\right)=\sum_{i=1}^{N} l\left(y_{i}, f\left(x_{i} ; \theta_{ctr}\right)\right)+\sum_{i=1}^{N} l\left(y_{i} \& z_{i}, f\left(x_{i} ; \theta_{ctr}\right) \times f\left(x_{i} ; \theta_{cvr}\right)\right)$$
其中， $\theta_{ctr}$和$\theta_{cvr}$分别是CTR网络和CVR网络的参数，$l(*)$是交叉熵损失函数。在CTR任务中，有点击行为的展现事件构成的样本标记为正样本，没有点击行为发生的展现事件标记为负样本；在CTCVR任务中，同时有点击和购买行为的展现事件标记为正样本，否则标记为负样本。

### 创新点
**共享Embedding层**
- CTR和CVR网络使用相同特征和特征embedding，即两者从Concatenate之后才学习各自部分独享的参数，这样能充分利用所有数据，缓解单独训练CVR的数据稀疏问题。

**隐式学习pCVR**
- pCVR（粉色节点）仅是网络中的一个variable，没有显示的监督信号，因为我们也没办法显式的给出真实的CVR。
- 这个可以从模型的损失函数中看出来，loss只和pCTR与pCTCVR相关，而pCTCVR是pCVR与pCTR相乘得到的，模型拟合了pCTR和pCTCVR，那么pCVR相当于隐含地被训练了，并且pCVR这块输出使用sigmoid激活的保证了值域稳定。

### 解决常规CVR预估的问题
**解决样本选择（BBS）问题**
* 全空间建模： 和CTR一样，在全部展现样本上建模。pCTCVR、pCTR和pCVR都定义在全样本空间。通过分别估算单独训练的模型pCTR和pCTCVR并通过关系式可以获得pCVR，三个关联的任务共同训练分类器，能够利用数据的序列模式并相互传递信息，保障物理意义。

**解决样本稀疏（DS）问题**
* 迁移学习：在ESMM中，CVR网络的Embedding参数与CTR任务共享，遵循特征表示迁移学习范式。Embedding Layer 将大规模稀疏输入映射到低维稠密向量中，主导深度网络参数。CTR任务所有展现样本规模比CVR任务要丰富多个量级，该参数共享机制使ESMM中的CVR网络可以在未点击展现样本中进行学习。

## 实验
由于ESMM模型创新性地利用了用户的序列行为做完模型的训练样本，因此并没有公开的数据集可供测试，阿里的技术同学从淘宝的日志中采样了一部分数据，作为[公开的测试集](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)。阿里妈妈的工程师们分别在公开的数据集和淘宝生产环境的数据集上做了测试，相对于其他几个主流的竞争模型，都取得了更好的性能。

### 对比实验
| 对照算法 | 描述 |
| --- | --- |
| BASE | 单CVR任务作为baseline |
| AMAN | 从未点击样本中随机抽样作为负例加入训练 |
| OVERSAMPLING | 对点击后的转化正样本过采样 |
| UNBIAS | 使用rejection sampling方式采样样本 |
| DIVISION | 训练CTR和CTCVR两个任务，除法运算得到pCVR |
| ESMM-NS | ESMM结构两个任务不共享Embedding |

### Comparison of different models on Public Dataset
![ESSM实验效果数据](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/ESMM实验效果数据.png)

与BASE模型相比，ESMM在CVR任务上实现了2.56％的AUC绝对值提升，这表明，即使对于有偏差的样本，它也具有良好的泛化性能。在具有完整样本的CTCVR任务上，它带来3.25％的AUC增益。这些结果验证了我们建模方法的有效性。

### Comparison of different models w.r.t. different sampling rates on Product Dataset
![](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/ESMM在生产环境中的效果.png)
与BASE模型相比，ESMM在CVR任务上实现2.18％的绝对AUC提升，在CTCVR任务上实现2.32％的绝对AUC提升。 对于工业应用而言，这是一项重大改进，在这种应用中，AUC提升高达0.1％


## 结论
1. ESMM提出了一种新颖的CVR建模方案，充分利用了用户操作的顺序模式。
2. 在CTR和CTCVR的两个辅助任务的帮助下，ESMM可以有效地解决实际实践中遇到的CVR建模的样本选择偏差和数据稀疏性的挑战。
3. ESMM是典型的share-bottom结构，即底层特征共享方式，在任务之间都比较相似或者相关性比较高的场景下能带来很好的效果。
4. ESMM可以看成一个MTL框架，其中子任务的网络结构可以替换，当中有很大的想象空间。

最后，引用一下朱小强对ESMM的[评价](https://zhuanlan.zhihu.com/p/54822778)
>![深度学习时代MTL建模范式](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/深度学习时代MTL建模范式.jpg)
关于ESMM模型多说两句，我们展示了对同态的CTR和CVR任务联合建模，帮助CVR子任务解决样本偏差与稀疏两个挑战。事实上这篇文章是我们总结DL时代Multi-Task Learning建模方法的一个具体示例。图中给出了更为一般的网络架构。据我所知这个工作在这个领域是最早的一批，但不唯一。今天很多团队都吸收了MTL的思路来进行建模优化，不过大部分都集中在传统的MTL体系，如研究怎么对参数进行共享、多个Loss之间怎么加权或者自动学习、哪些Task可以用来联合学习等等。ESMM模型的特别之处在于我们额外关注了任务的Label域信息，通过展现>点击>购买所构成的行为链，巧妙地构建了multi-target概率连乘通路。传统MTL中多个task大都是隐式地共享信息、任务本身独立建模，ESMM细腻地捕捉了契合领域问题的任务间显式关系，从feature到label全面利用起来。这个角度对互联网行为建模是一个比较有效的模式，后续我们还会有进一步的工作来推进。

## 参考文献
[1][Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf)
[2][CVR预估的新思路：完整空间多任务模型/杨旭东/知乎](https://zhuanlan.zhihu.com/p/37562283)
[3][镶嵌在互联网技术上的明珠：漫谈深度学习时代点击率预估技术进展/朱小强/知乎](https://zhuanlan.zhihu.com/p/54822778)
[4][ESMM-完整空间下的多任务学习/知天易or逆天难/CSDN](https://blog.csdn.net/u013019431/article/details/100027405)
[5][ESMM模型笔记/巴拉巴拉朵/CSDN](https://blog.csdn.net/whgyxy/article/details/108565089)