---
title: ChatGPT模型的三层理解
date: 2023-02-24 10:00:00
updated: 2023-02-24
categories:
- 机器学习
tags:
- ChatGPT
- InstructGPT
keywords: ChatGPT,InstructGPT
description: ChatGPT模型的三层理解，1）训练流程：白话讲解模型是怎么训练出来的；2）模型实现：基于论文讲解模型训练的细节；3）发展脉络：从模型演进的视角看其创新点和贡献。
---

目前chatGPT的论文还没有公布，官方介绍里面讲：“ChatGPT is a sibling model to InstructGPT”，对比ChatGPT官网的模型训练流程和InstructGPT的流程图，基本是一致的，只是ChatGPT基于GPT-3.5进行的训练，我们下面的讲解就暂时以InstructGPT代替ChatGPT展开。

![ChatGPT vs InstructGPT](https://oss.imzhanghao.com/img/202302240527749.png)


## 第一层：训练流程
> 白话讲解ChatGPT的训练流程，不涉及模型训练细节，主要讲解整个训练是如何完成的。

ChatGPT学习的四阶段

### 1.1 第一步：学习文字接龙

![GPT文字接龙](https://oss.imzhanghao.com/img/202302141132504.png)
GPT（Generative Pre-trained Transformer）是一个会**文字接龙**的模型，给他一段文本，他会预测下一个字是什么。

![样本生成逻辑](https://oss.imzhanghao.com/img/202302141133884.png)
训练一个文字接龙的模型是**不需要人工标注的文本的**，只需要在网上收集大量的文字，就可以学文字接龙这件事情。

![真实的输出](https://oss.imzhanghao.com/img/202302141131855.png)
GPT真实的输出是一个概率分布，“你好”的输入，可能跟“高”、“美”、“吗”等等词，每一次的输出都是不同的，出现的概率也不一样。

![文字接龙回答问题](https://oss.imzhanghao.com/img/202302141155611.png)
文字接龙已经可以用来回答问题了，但是...

![每次的输出都不同](https://oss.imzhanghao.com/img/202302141153648.png)

GPT输出的是一个概率分布，后面可以接各式各样的句子，很多并不是我们想要的。
那我们如何引导GPT产生有用的输出呢？

### 1.2 第二步：人类老师引导文字接龙方向
找人来思考想问GPT的问题，并人工提供正确答案。

- 西安的地标是什么？ --> 钟楼
- 如何学习深度学习？ -->  需要先知道基本概念...
- 请把这句话做翻译...

让原始的GPT模型在这部分质量较高的数据集上学习，多看一些有益的句子，期待他能产生出有用的输出。这里不需要穷尽所有问题，只需要告诉GPT人类的偏好。

### 1.3 第三步：模仿人类老师的喜好
训练一个模仿老师的模型，学习人类老师评分高低的标准。
![模仿人类老师的喜好](https://oss.imzhanghao.com/img/202302141337882.png)

如果人类提交的是“钟楼”这个答案好于“谁来告诉我呀”，那么Teacher Model给“钟楼”这个的打分就要比“谁来告诉我呀”的打分高。

### 1.4 第四步：用强化学习向模拟老师学习
把“接龙模型GPT”和“老师模型Teacher Model”组合起来使用。

![强化学习](https://oss.imzhanghao.com/img/202302141407380.png)

Teacher Model通过前面的学习已经学到，如果答案是一个问句，它不是一个好的答案，给予低分。这个Teacher Model输出的低分就是强化学习的奖励Reward，强化学习通过调整参数，得到最大的Reward

![强化学习](https://oss.imzhanghao.com/img/202302141408181.png)
经过强化学习以后，GPT就变成了ChatGPT，能够输出我们想要的答案了。

> 总结：整个过程就是教GPT从“**想说什么就说什么**”到“**说人类想要他说的**”。

## 第二层：工程实现
> 重点讲解Instruct GPT的论文，《Training language models to follow instructions with human feedback》(训练语言模型是他能够服从人类的指示)。68页的论文主要内容在讲工程实现，包括怎么挑选合适的标注人员，各个模型都是如何准备训练数据的，甚至还给了很多标注表格的模板和训练样本范例。

大型语言模型中的一致性问题通常表现为：
- **提供无效帮助**：没有遵循用户的明确指示。
- **内容胡编乱造**：虚构不存在或错误事实的模型。
- **缺乏可解释性**：人们很难理解模型是如何得出特定决策或预测的。
- **内容偏见有害**：一个基于有偏见、有害数据训练的语言模型可能会在其输出中出现这种情况，即使它没有明确指示这样做。

![three steps of Instruct GPT](https://oss.imzhanghao.com/img/202302141453190.png)

InstructGPT/ChatGPT的提出了一种利用人类反馈来解决这一问题的方案，方法总体上可以分成3步：
- 根据采集的SFT数据集对GPT-3进行有监督的微调（Supervised FineTune，SFT）；
- 收集人工标注的对比数据，训练奖励模型（Reword Model，RM）；
- 使用RM作为强化学习的优化目标，利用PPO算法微调SFT模型。

### 2.1 数据来源
训练数据有两个来源：
- 由我们的标注人员编写的Prompt数据集
- 提交给早期 InstructGPT模型版本API的Prompt数据集

**标记人员**

OpenAI通过一系列的筛选，找到了**40**个对不同人口群体的偏好敏感并且善于识别可能有害的输出的**全职**标记人员。整个过程中，OpenAI的研发人员跟这些标记人员紧密合作，给他们进行了培训，并对他们是否能够代表大众的偏好进行了评测。

标记人员的工作是根据内容自己编写prompt，并且要求编写的Prompt满足下面三点：
- **简单任务**：标记人员给出任意简单的任务，同时要确保任务的多样性；
- **Few-shot任务**：标注人员写出一个指示，同时写出其各种不同说法。
- **用户相关的**：从接口中获取用例，然后让标记人员根据这些用例编写prompt。

![标记人员分布](https://oss.imzhanghao.com/img/202302161318246.png)

40名外包员工来自美国和东南亚，分布比较集中且人数较少， InstructGPT/ChatGPT的目标是训练一个价值观正确的预训练模型，它的价值观是由这40个外包员工的价值观组合而成。而这个比较窄的分布可能会生成一些其他地区比较在意的歧视，偏见问题。

**API用户**

OpenAI训练了一个早期版本的InstructGPT，开放给了一部分用户，根据他们提问信息构造样本，对数据集做了如下操作：
- 删除了一些重复的、包含个人信息的prompt；
- 每个用户只取200条prompt；
- 按照用户ID划分训练集、验证集和测试集，避免类似问题同时出现在训练集和验证集。

![API prompt dataset](https://oss.imzhanghao.com/img/202302161135589.png)

prompt种类共有9种，而且绝大多数是生成类任务，可能会导致模型有覆盖不到的任务类型；


### 2.2 训练数据
基于上面的数据集，生成了三份不同的训练数据集：
|  数据集  | 生成方法  |  prompt量  | 训练任务  |
|  ----  | ----  |----  | ----  |
| SFT dataset  | 由标记人员编写prompt对应的回答 | 13k | Supervised FineTune(SFT) |
| RM dataset  | 由标记人员对gpt产生的答案进行质量排序 | 33k | Reword Model(RM) |
| PPO dataset | 不需要人工参与，gpt产生结果，RM进行打分 | 31k  | Reinforcement learning (RL) |

![prompt](https://oss.imzhanghao.com/img/202302161137916.png)

数据中96%以上是英文，其它20个语种例如中文，法语，西班牙语等加起来不到4%，这可能导致InstructGPT/ChatGPT能进行其它语种的生成时，效果应该远不如英文；

### 2.3 模型微调Supervised fine-tuning（SFT）
![神经网络可视化](https://oss.imzhanghao.com/img/202302161333564.png)
低层的网络主要学习图像的边缘或色斑，中层的网络主要学习物体的局部和纹理，高层的网络识别抽象的语义。

由上面的案例可知，我们可以把一个神经网络分成两块：1）低层的网络进行特征抽取，将原始信息变成容易被后面任务使用的特征；2）输出层的网络进行具体任务的预测。输出层因为涉及到具体任务没办法在不同任务中复用，但是低层的网络是具有通用型的，可以应用到其他任务上。

![微调](https://oss.imzhanghao.com/img/202302161321218.png)
微调由以下4步构成：
- 在源数据集上**预训练**一个神经网络模型，即源模型。
- 创建一个新的神经网络模型，即目标模型。它**复制**了源模型上除了输出层外的所有模型设计及其参数。我们假设这些模型参数包含了源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层与源数据集的标签紧密相关，因此在目标模型中不予采用。
- 为目标模型添加一个输出大小为目标数据集类别个数的**输出层**，并随机初始化该层的模型参数。
- 在目标数据集上**训练目标模型**。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。

![SFT](https://oss.imzhanghao.com/img/202302170628736.png)

这里的SFT就是基于训练好的GPT-3模型进行的微调，在数据集上面进行了16个epoch的训练，数据量比较小，模型比较大，一个epoch以后就过拟合了，但是论文里面说这个没关系，因为他不是直接拿出去用，而是用来初始化后面的模型。

### 2.4 训练奖励模型Reward modeling（RM）
一个Response不能给具体的打分值，只能说一个Response比另一个Response更好或者差不多，所以训练奖励模型的数据是一个标注人员根据生成结果排序的形式，它可以看做一个回归模型。

![pairwise](https://oss.imzhanghao.com/img/202302161450605.png)
奖励模型使用的是PairWise的训练方法，PairWise的基本思路是对样本进行两两比较，构建偏序文档对，从比较中学习顺序。PairWise就是希望通过正确估计一对文档的顺序，而得到整体的正确顺序，比如一个正确的排序为：“A>B>C”，PairWise通过学习两两之间的关系“A>B”，“B>C”和“A>C”来推断“A>B>C”。

![RM](https://oss.imzhanghao.com/img/202302170636619.png)
奖励模型的结构是将SFT训练后的模型的最后的嵌入层去掉后的模型。它的输入是prompt和Reponse，输出是奖励值。GPT-3最大的模型时175B的，但是发现这种规模的模型训练不稳定，最后才用了6B的版本。

### 2.5 强化学习Reinforcement learning（RL）

![强化学习](https://oss.imzhanghao.com/img/202302161741774.png)
强化学习（Reinforcement learning，RL）讨论的问题是一个智能体（agent）怎么在一个复杂不确定的 环境（environment) 里面去极大化它能获得的奖励。通过感知所处环境的 状态（state) 对 动作（action） 的反应（reward），来指导更好的动作，从而获得最大的收益（return），这被称为在交互中学习，这样的学习方法就被称作强化学习。

![PPO](https://oss.imzhanghao.com/img/202302161736793.png)

PPO（Proximal Policy Optimization）近端策略优化算法，是一种典型的强化学习算法。它通过观测信息选出一个行为直接进行反向传播，当然出人意料的是他并没有误差，而是利用reward奖励直接对选择行为的可能性进行增强和减弱，好的行为会被增加下一次被选中的概率，不好的行为会被减弱下次被选中的概率。

![RL](https://oss.imzhanghao.com/img/202302170633404.png)
这里我们随机取一个用户的prompt，然后使用SFT模型生成一个response，使用奖励模型对当前的prompt-response对打分作为Reward。

### 2.6 结论
![结论](https://oss.imzhanghao.com/img/202302161926570.png)

该模型基于三个标准进行评估：
- **帮助性**：判断模型遵循用户指示以及推断指示的能力。
- **真实性**：判断模型在封闭领域任务中有产生虚构事实的倾向。
- **无害性**：标注者评估模型的输出是否适当、是否包含歧视性内容。

跟SFT 175B的模型对比效果，所以我们可以看到SFT 175B的胜率是0.5。

在对我们的prompt进行人工评估过程中，1.3B个参数的InstructGPT模型的输出优于 175B的GPT-3模型的输出，尽管参数少100倍。此外，InstructGPT模型显示了真实性的提高和有毒输出生成的减少，同时在公共数据集上的性能下降却很小。尽管InstructGPT 仍然会犯一些简单的错误，但我们的结果表明，**根据人类反馈进行微调是使语言模型与人类意图保持一致的一个有前途的方向**。


## 第三层：发展脉络
> 任何模型的突破和成功，都不是一蹴而就的，是不断的学习前人的经验和结论，自己一步一步走出来的，这个需要的是厚积薄发。ChatGPT也是一样，要真正理解它，就需要看它之前的工作，从发展脉络的角度去看问题。

![模型演进](https://oss.imzhanghao.com/img/202302240534919.png)

### 3.1 Encoder-Decoder
Cho在2014年的《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中提出了Encoder–Decoder结构，它由两个RNN组成，另外本文还提出了GRU的门结构，相比LSTM更加简洁，而且效果不输LSTM。

![Encoder-Decoder框架](https://oss.imzhanghao.com/img/20210526143504.png)

生成目标句子单词的过程成了下面的形式，其中f1是Decoder的非线性变换函数：
$$\begin{array}{l}
\mathbf{Y}_{\mathbf{1}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}\right) \\
\mathbf{Y}_{2}=\mathbf{f} \mathbf{1}\left(\mathbf{C}, \mathbf{Y}_{\mathbf{1}}\right) \\
\mathbf{Y}_{3}=\mathbf{f} \mathbf{1}\left(\mathbf{C}, \mathbf{Y}_{\mathbf{1}}, \mathbf{Y}_{2}\right)
\end{array}$$

Encoder和Decoder部分可以是任意的文字、语音、图像和视频数据，模型可以采用CNN，RNN，BiRNN、LSTM、GRU等等，所以基于Encoder-Decoder的结构，我们可以设计出各种各样的应用算法。比如：1）**文字-文字**：机器翻译，对话机器人，文章摘要，代码补全； 2） **音频-文字**：语音识别； 3） **图片-文字**：图像描述生成

> Encoder-Decoder的出现，对于很多领域的影响是非常深远的，比如机器翻译的任务，再也不需要做词性分析、词典查询、语序调整和隐马尔可夫模型等等方案，几乎都是基于Encoder-Decoder框架的神经网络方案。

### 3.2 Attention机制
Attention机制最早在视觉领域提出，2014年Google Mind发表了《Recurrent Models of Visual Attention》，使Attention机制流行起来，这篇论文采用了RNN模型，并加入了Attention机制来进行图像的分类。

2015年，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中，将attention机制首次应用在nlp领域，其采用Seq2Seq+Attention模型来进行机器翻译，并且得到了效果的提升。

![人类的视觉注意力](https://oss.imzhanghao.com/img/20210526141037.png)

语义编码C是由句子Source的每个单词经过Encoder编码产生的，这意味着不论是生成哪个单词，其实句子Source中任意单词对生成某个目标单词来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

我们拿机器翻译来解释一下注意力在Encoder-Decoder模型中的作用就更好理解了，比如输入的是英文句子：Tom chase Jerry，Encoder-Decoder框架逐步生成中文单词：“汤姆”，“追逐”，“杰瑞”。

在翻译“杰瑞”这个中文单词的时候，没有注意力的模型里面的每个英文单词对于翻译目标单词“杰瑞”贡献是相同的，很明显这里不太合理，显然“Jerry”对于翻译成“杰瑞”更重要，但是没有注意力的模型是无法体现这一点的。

目标句子中的每个单词都应该学会其对应的源语句子中单词的注意力分配概率信息。这意味着在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的Ci。理解Attention模型的关键就是这里，即**由固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的Ci**。增加了注意力模型的Encoder-Decoder框架理解起来如图所示。

![引入注意力模型的Encoder-Decoder框架](https://oss.imzhanghao.com/img/20210526150157.png)

即生成目标句子单词的过程成了下面的形式：
$$\begin{array}{l}
\mathbf{Y}_{\mathbf{1}}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{\mathbf{1}}\right) \\
\mathbf{Y}_{2}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{2}, \mathbf{Y}_{\mathbf{1}}\right) \\
\mathbf{Y}_{3}=\mathbf{f} \mathbf{1}\left(\mathbf{C}_{3}, \mathbf{Y}_{\mathbf{1}}, \mathbf{Y}_{2}\right)
\end{array}$$

![attention机制添加以后的翻译的过程](https://oss.imzhanghao.com/img/202302240536840.png)

> Attention机制由来已久，真正让其大放异彩的，是后来Google的一篇文章《Attention is All You Need》。

### 3.3 Transformer
2017年，Google机器翻译团队发表的《Attention is All You Need》中，提出了他们的Transformer架构，Transformer基于经典的机器翻译Seq2Seq框架，完全抛弃了RNN和CNN等网络结构，而仅仅采用Attention机制来进行机器翻译任务，在WMT 2014的数据集上取得了很好的成绩。

![Transformer的架构](https://oss.imzhanghao.com/img/202109290538512.png)

**编码器**
编码器由N=6个相同的layer组成，layer指的就是上图左侧的单元，最左边有个“Nx”，这里是x6个。每个Layer由两个子层（Sub-Layer）组成,第一个子层是Multi-head Self-attention Mechanism，第二个子层比较简单，是Fully Connected Feed-Forward Network。其中每个子层都加了残差连接（Residual Connection）和层归一化（Layer Normalisation），因此可以将子层的输出表示为：$\text { LayerNorm }(x+\operatorname{SubLayer}(x))$

**解码器**
解码器同样由N=6个相同layer组成，因为编码器是并行计算一次性将结果直接输出，而解码器是一个词一个词输入，所以解码器除了每个编码器层中的两个子层之外，还插入第三子层，其对编码器堆栈的输出执行multi-head attention。每个子层也都加了残差连接（Residual Connection）和层归一化（Layer Normalisation）。解码器中对self-attention子层进行了修改，以防止引入当前时刻的后续时刻输入，这种屏蔽与输出嵌入偏移一个位置的事实相结合，确保了位置i的预测仅依赖于小于i的位置处的已知输出。

> Transformer出现后，开始取代RNN（循环神经网络）和 CNN（卷积神经网络）成为最热门的信息提取工具，以前做NLP（自然语言处理）研究的喜欢用RNN，做CV（计算机视觉）研究的喜欢用CNN，现在大家都用Transformer了，很多研究进展可以同步到另一个领域。

### 3.4 GPT
传统的NLP模型往往使用大量的数据对有监督的模型进行任务相关的模型训练，但是高质量的标注数据往往很难获得，而且不同任务的模型很难泛化到其他任务上，所以OpenAI在《Imporoving Language Understanding By Generative Pre-training》提出先在大量的无标签数据上训练一个语言模型，然后再在下游具体任务的有标签数据集上进行fine-tune。

GPT是典型的**预训练+微调**的两阶段模型。预训练阶段就是用海量的文本数据通过无监督学习的方式来获取语言学知识，而微调就是用下游任务的训练数据来获得特定任务的模型。

举一个例子来形容预训练和微调的关系，我们从幼儿园到高中就像预训练过程，不断学习知识，这些知识包罗万象包括语数英物化生等等，最重要的特征就是预训练模型具有很好的通用性；然后读大学的时候需要确定一个专业方向作为未来的职业，所以就会去重点学习专业知识，从而让我们成为更加适合某个专业方向的人才，最重要的特征就是具有极强的专业性。

![gpt使用的是transformer的Decoder部分](https://oss.imzhanghao.com/img/202302171642998.png)

GPT模型效果还是非常出色的，12个任务数据集中9个达到了最好效果。

### 3.5 BERT
2018年10月，Google发表《BERT_Pre-training of Deep Bidirectional Transformers for Language Understanding》，提出了类似于GPT的预训练模型BERT。

![bert使用的是transformer的Encoder部分](https://oss.imzhanghao.com/img/202302171650512.png)

BERT和GPT采用了不同的技术路线，简单理解，BERT是一个双向模型，可以联系上下文进行分析，更擅长**完形填空**；而GPT是一个单项模型，只能从左到右进行阅读，更擅长**写作文**。

发布更早的GPT-1输给了晚4个月发布的BERT，而且是完败。在当时的竞赛排行榜上，阅读理解领域已经被BERT屠榜了。此后，BERT也成为了NLP领域最常用的模型。
![bert on GLUE](https://oss.imzhanghao.com/img/202302181424811.png)

### 3.6 GPT-2 & GPT-3
OpenAI既没有认输，也非常头铁。虽然GPT-1效果不如BERT，但OpenAI没有改变策略，而是坚持走大模型路线。接下来的两年（2019、2020年），在几乎没有改变模型架构的基础上，OpenAI 陆续推出参数更大的迭代版本GPT-2、GPT-3，前者有15亿参数，后者有1750亿参数，最终GPT-3的模型效果震惊世人，成功出圈。

|  模型  | 发布时间  |  层数  | 头数  |词向量长度  |参数量  |预训练数据量  |
|  ----  | ----  |----  | ----  | ----  |----  | ----  |
|  GPT-1  | 2018 年 6 月  |  12  | 768  | 768  | 1.17 亿  |约5GB  |
|  GPT-2  | 2019 年 2 月  |  48  | -  |1600  | 15 亿  |40GB  |
|  GPT-3  | 2020 年 5 月  |  96  | 96  |12888  | 1750 亿  |45TB  |

但是单纯的说用一个更大的模型打败了对手感觉还不够，GPT-2和3开始卷另一个方向，不用做梯度更新和微调依然可以完成很多任务，也就是Zero-shot。在OpenAI眼中，未来的通用人工智能应该长这个样子：**有一个任务无关的超大型的语言模型（Large Language Model, LLM），用来从海量数据中学习各种知识，这个LLM以生成一切的方式，来解决各种各样的实际问题，而且它应该能听懂人类的命令，以便于人类使用。**

![Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning.](https://oss.imzhanghao.com/img/202302181434145.png)

> 本身在这么大的模型上做梯度更新和微调就是一个门槛很高的事情，GPT-3模型对于Zero-shot、one-shot和few-shot的支持，使得基于他可以衍生出了上百个基于GPT-3的应用。

### 3.7 instruction Tuning
Google在2021年发表《Finetuned Language Models are Zero-Shot Learners》，提出了一种提高语言模型Zero-shot能力的方法。

比如我们要进行一句话的情感分析，“带女朋友去了一家餐厅，她吃的很开心”，我们可以有两种做法：
- 利用模型补全的能力，改造prompt：“带女朋友去了一家餐厅，她吃的很开心，这家餐厅太__了！”
- 直接告诉模型任务：“判断这句话的情感：带女朋友去了一家餐厅，她吃的很开心。选项：A=好，B=一般，C=差”
instruction Tuning是第二种。

![pretrain–finetune and prompting and instruction Tuning](https://oss.imzhanghao.com/img/202302201150042.png)

Instruction tuning的动机是为了提高语言模型对instructions的**理解和响应能力**。其想法是，通过监督来教语言模型执行instructions描述的任务，从而使它将学习到如何遵循instructions，当面对unseen的任务时，模型自然而然地就会遵循instruction做出响应。
![instruction-tuning](https://oss.imzhanghao.com/img/202302201152876.png)
首先在包括commonsense reasoning、machine translation、sentiment analysis等NLP task上进行微调，然后在从未见过的natural language inference任务上进行zero-shot evaluation

![performance of zero-shot FLAN](https://oss.imzhanghao.com/img/202302201404574.png)

### 3.8 InstructGPT/ChatGPT
InstructGPT论文《Training language models to follow instructions with human feedback》，我们已经讲清楚language models和instructions，剩下就是with human feedback。

人工反馈的强化学习（Reinforcement Learning from Human Feedback，RLHF）参考了OpenAI前面的两个工作。

Ziegler在2019年的《Fine-Tuning Language Models from Human Preferences》
![Ziegler et al. (2019)](https://oss.imzhanghao.com/img/202302201425975.png)

Stiennon在2020年《Learning to summarize from human feedback》
![Stiennon et al. (2020),](https://oss.imzhanghao.com/img/202302201426679.png)

ChatGPT和InstructGPT在模型结构，训练方式上都完全一致，即都使用了指示学习（Instruction Learning）和人工反馈的强化学习（Reinforcement Learning from Human Feedback，RLHF）来指导模型的训练，它们不同的仅仅是采集数据的方式上有所差异。

## 写在最后

### 4.1 安全性
2016年，微软AI Tay，种族歧视，下线。
Microsoft shuts down AI chatbot after it turned into a Nazi。
https://www.cbsnews.com/news/microsoft-shuts-down-ai-chatbot-after-it-turned-into-racist-nazi/

2021年，Facebook，AI将黑人标上了灵长目的标签
Facebook Apologizes After A.I. Puts ‘Primates’ Label on Video of Black Men
Facebook called it “an unacceptable error.” The company has struggled with other issues related to race.
https://www.nytimes.com/2021/09/03/technology/facebook-ai-race-primates.html

语言模型的输出特别灵活，导致出错的概率会更大，OpenAI作为一个创业公司，媒体对于GPT的容忍度大一些，如果是大公司做的GPT模型，可能已经下架了。

### 4.2 成本
![LLM模型](https://oss.imzhanghao.com/img/202302240540923.png)

大模型背后离不开大数据、大算力。GPT-2用于训练的数据取自于Reddit上高赞的文章，数据集共有约800万篇文章，累计体积约40G；GPT-3模型的神经网络是在超过45TB的文本上进行训练的，数据相当于整个维基百科英文版的160倍。

在算力方面，GPT-3.5在微软Azure AI超算基础设施（由V100GPU组成的高带宽集群）上进行训练，总算力消耗约3640PF-days（即每秒一千万亿次计算，运行3640天）。可以说，大模型的训练就是靠烧钱烧出来的。据估算，OpenAI的模型训练成本高达1200万美元，GPT-3的单次训练成本高达460万美元。

根据《财富》杂志报道的数据，2022年OpenAI的收入为3000万美元的收入，但净亏损总额预计为5.445亿美元。阿尔特曼在Twitter上回答马斯克的问题时表示，在用户与ChatGPT的每次交互中OpenAI花费的计算成本为“个位数美分”，随着ChatGPT变得流行，每月的计算成本可能达到数百万美元。

大模型高昂的训练成本让普通创业公司难以为继，因此参与者基本都是的科技巨头。


## 参考资料
- [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)
- [Instruct GPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [GPT-1_Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2_Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3_Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Instruct Learning_Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
