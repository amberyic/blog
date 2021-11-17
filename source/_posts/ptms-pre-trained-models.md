---
title: 自然语言处理预训练技术综述
date: 2021-11-15
updated: 2021-11-15
categories:
- 机器学习
tags:
- 预训练
- 自然语言处理
keywords: 预训练,PTM,Pre-Trained,Fine-tuning,NNLM,word2vec,Glove,ELMo,GPT,Bert
description: 本文梳理预训练技术的原理和发展脉络，着重讲解了几个具有代表性的模型，第一代的预训练模型：NNLM,word2vec,Glove，和第二代的预训练模型：ELMo,GPT,Bert。这是一个正在井喷的研究方向，简单描述了目前预训练技术的几个延伸方向以及应用到下游任务的方案。
---
## 预训练
预训练(Pre-trained Models,PTMs)的实施过程跟**迁移学习**是一样的，一般是先在一个基础数据集上进行任务训练，生成一个基础网络，然后将学习到的特征重新进行微调或者迁移到另一个目标网络上，用来训练新目标任务。

预训练是在大量常规数据集上学习数据中的“**共性**”，然后在特定领域的少量标注数据学习“**特性**”，这样子模型只需要从“共性”出发，去学习特定任务的“特性”部分即可。

这和小孩子读书一样，一开始语文、数学、化学都学，读书、网上游戏等，在脑子里积攒了很多。当他学习计算机时，实际上把他以前学到的所有知识都带进去了。如果他以前没上过中学，没上过小学，突然学计算机就不懂这里有什么道理。**预训练模型就意味着把人类的语言知识，先学了一个东西，然后再代入到某个具体任务，就顺手了，就是这么一个简单的道理。**

### 为什么需要预训练
- 预训练模型中的参数都是从大量数据中训练得来，比起在自己的数据集上从头开始训练参数，在预训练模型参数基础上继续训练的方式肯定要快一些。
- 预训练模型是通过海量数据训练得来，更好地学到了数据中的普遍特征，比起在自己的数据集上从头开始训练参数，使用预训练模型参数通常会有更好的泛化效果。

### 计算机视觉上的预训练
预训练首先是在计算机视觉方向取得较好效果并实现大规模应用的，我们会在庞大的ImageNet语料库上预训练模型，然后针对不同的任务在较小的数据上进一步微调。这比随机初始化要好得多，因为模型学习了一般的图像特征，然后可以将其用于各种视觉任务。
ImageNet这个数据集，数据量足够大，而且分类齐全，不限定领域，具有很好的通用型，使用范式一般如下图所示：
![ImageNet预训练](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111041801550.png)

### 自然语言处理上的预训练
借鉴视觉领域的做法,自然语言处理领域开始尝试使用预训练技术实现迁移学习，但是预训练在自然语言处理领域大爆发会缓慢很多，主要是因为自然语言处理任务(除机器翻译)没有计算机视觉方面那么多的标注好的数据集，而且没有很好的特征提取器，直到最近几年几个关键技术的成熟，神经网络才开始全面引入到了自然语言理解。从大规模的语言数据到强有力的算力，加上深度学习，把整个自然语言带到一个新的阶段。

自然语言处理预训练在不同时期有不同的称谓，但是，**本质是使用大量语料预测相应单词或词组，生成一个半成品用以训练后续任务**。

自然语言处理任务可以分为以下3个模块:**数据处理、文本表征和特定任务模型**。其中，**数据处理模块**和**特定任务模型模块**需要根据具体任务的不同做相应设计，而**文本表征模块**则可以作为一个相对**通用**的模块来使用。类似于计算机视觉领域中基于ImageNet预训练模型的做法，自然语言处理领域也可以预训练一个通用的文本表征模块，这种通用的文本表征模块对于文本的迁移学习具有重要意义。

### 发展历史
自然语言处理的预训练方法属于**自然语言的表示学习**，自然语言表示学习的形成已经经过了长期的历史发展。

- 1948年N-gram分布式模型被提出来，使用one-hot对单词进行编码，这是最初的语言模型，存在维度灾难和语义鸿沟等问题。
- 1986年出现了分布式语义表示，即用一个词的上下文来表示该词的词义，他在one-hot的基础上压缩了描述语料库的维度，从原先的V-dim降低为了自己设定的K值。当时通用的方案是基于向量空间模型（Vector Space Model，VSM）的**词袋假说**（Bag of Words Hypothesis），即一篇文档的词频（而不是词序）代表了文档的主题，我们可以构造一个term-document矩阵，提取行向量做为word的语义向量，或者提取列向量作为文档的主题向量，使用奇异值分解(SVD)的进行计算。
- 2003年经典的NNLM神经语言模型被提出，开始使用神经网络来进行语言建模。更早期百度 IDL（深度学习研究院）的徐伟在他2000年发表的文章《Can Artificial Neural Networks Learn Language Models?》中也有相似方向的探索。
- 2013年word2vec被提出并在NLP领域大获成功，他基于向量空间模型的**分布假说**（Distributional Hypothesis），即上下文环境相似的两个词有着相近的语义，构造一个word-context的矩阵，矩阵的列变成了context里的word，矩阵的元素也变成了一个context窗口里word的共现次数。Word Embedding是Word2Vec模型的中间产物，是在不断最小化损失函数时候，不断迭代更新生成的。
- 2018年出现了预训练语言模型。

### 传统的预训练技术 VS 神经网络预训练技术
**传统的预训练技术**
传统预训练技术与模型耦合较为紧密，该技术与模型之间并没有明确的区分界限，为了方便阐述，将语料送入模型到生成词向量的这一过程称为传统预训练技术。
![传统的预训练技术](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111120845409.png)

**神经网络预训练技术**
神经网络预训练技术是在预训练阶段采用神经网络模型进行预训练的技术统称，由于预训练与后续任务耦合性不强，能单独成为一个模型，因此也称为预训练语言模型，这一称谓是区别于传统预训练技术的叫法。

神经网络自然语言处理的预训练发展经历从浅层的词嵌入到深层编码两个阶段，按照这两个主要的发展阶段，我们归纳出预训练的两大范式：「浅层词嵌入」和「上下文的词嵌入」。
- **第一代预训练旨在学习浅层词嵌入(Word Embeddings)**。由于下游的任务不再需要这些模型的帮助，因此为了计算效率，它们通常采用浅层模型，如 Skip-Gram 和 GloVe。尽管这些经过预训练的嵌入向量也可以捕捉单词的语义，但它们却不受上下文限制，只是简单地学习「共现词频」。这样的方法明显无法理解更高层次的文本概念，如句法结构、语义角色、指代等等。
- **第二代预训练专注于学习上下文的词嵌入(Contextual Embeddings)**，如CoVe、ELMo、GPT以及BERT。它们会学习更合理的词表征，这些表征囊括了词的上下文信息，可以用于问答系统、机器翻译等后续任务。另一层面，这些模型还提出了各种语言任务来训练，以便支持更广泛的应用，因此它们也可以称为预训练语言模型。


本文重点讲解基于**神经网络**模型在**自然语言处理**领域的**预训练技术**。

## 关键技术
### Transfromer
Google 2017年提出了Transformer模型，之后席卷了整个NLP领域，红极一时的BERT、GPT-2都采用了基于Transformer的架构，现在都用到CV领域了，用于目标检测和全景分割的DETR就是代表。Transfromer的特征提取能力显著强于以往常用的CNN和RNN，**这可以让我们更快更好的在样本上学习知识**

Transformer之所以表现优异有以下几点原因：
- 模型并行度高，使得训练时间大幅度降低。
- 可以直接捕获序列中的长距离依赖关系。
- 可以产生更具可解释性的模型。

想详细了解Transfromer，可以参考我以前的文章[《Attention Is All You Need -- Transformer》](https://imzhanghao.com/2021/09/18/transformer/)

### 自监督学习
自监督学习是无监督学习的一种特殊方式，这些自监督的方法的核心是一个叫做 “pretext task” 的框架，它允许我们使用数据本身来生成标签，并使用监督的方法来解决非监督的问题。NLP预训练模型，就是利用自监督学习实现的，可以看做是一种去噪自编码器denoising Auto-Encoder。**这可以让我们在大规模无标注数据集上学习知识。**

在预训练模型中，最常用的自监督学习方法是自回归语言模型（AutoRegressive LM，AR）和自编码语言模型（AutoEncoder LM，AE）。 **自回归语言模型**根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词。 **自编码语言模型**根据上下文内容预测随机Mask掉的一些单词。

### 微调
微调旨在利用其标注样本对预训练网络的参数进行调整，可以将预训练的模型结果在新的任务上利用起来。
![微调](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111140618731.png)


## 第一代技术预训练技术：Word Embeddings

### NNLM
神经网络语言模型(Neural Network Language Model，NNLM)是2003年蒙特利尔大学的Yoshua Bengio教授在《A Neural Probabilistic Language Model》中提出来的模型，这个模型第一次用神经网络来解决语言模型的问题，虽然在当时并没有得到太多的重视，却为后来深度学习在解决语言模型问题甚至很多别的nlp问题时奠定了坚实的基础，后人站在Yoshua Bengio的肩膀上，做出了更多的成就。
![NNLM](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111101654276.png)
模型一共三层，第一层是**映射层**，将n个单词映射为对应word embeddings的拼接，其实这一层就是MLP的输入层；第二层是**隐藏层**，激活函数用tanh；第三层是**输出层**，因为是语言模型，需要根据前n个单词预测下一个单词，所以是一个多分类器，用softmax。整个模型最大的计算量集中在最后一层上，因为一般来说词汇表都很大，需要计算每个单词的条件概率，是整个模型的计算瓶颈。

**评价**
- NNLM模型是第一次使用神经网络对语言建模
- 由于模型使用的是全连接神经网络，所以只能处理定长序列。
- 由于模型最后一层使用softmax进行计算，参数空间巨大，训练速度极慢。

### Word2Vec
Word2Vec是从大量文本语料中以无监督的方式学习**语义知识**的一种模型，将单词从原先所属的空间**映射**到新的多维空间中，即把原先词所在空间嵌入(Embedding)到一个新的空间中去，用词向量的方式表征词的语义信息，通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。

Word2Vec模型中，主要有Skip-Gram和CBOW两种模型，从直观上理解，Skip-Gram是给定input word来预测上下文。而CBOW是给定上下文，来预测input word。
![CBOW&Skip-Gram](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111101903380.png)

**评价**
- 优化了计算效率，特别是google同时开源了工具包，使得其在工业界能够大规模使用。
- Word2vec并没有考虑到词序信息以及全局的统计信息等

### GloVe
Glove(Global Vectors for Word Representation)是一种无监督的词嵌入方法，该模型用到了语料库的全局特征，即单词的共现频次矩阵，来学习词表征（word representation）。

**第一步统计共现矩阵**：下面给出了三句话，假设这就是我们全部的语料。我们使用一个size=1的窗口，对每句话依次进行滑动，相当于只统计紧邻的词。这样就可以得到一个共现矩阵。共现矩阵的每一列，自然可以当做这个词的一个向量表示。这样的表示明显优于one-hot表示，因为它的每一维都有含义——共现次数，因此这样的向量表示可以求词语之间的相似度。
![共现矩阵](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111111038802.png)

**第二步训练词向量**：共现矩阵维度是词汇量的大小，维度是很大的，并且也存在过于稀疏的问题，这里我们使用**SVD矩阵分解**来进降维。
![SVD求解](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111110902602.png)

**评价**
- 利用词共现矩阵，词向量能够充分考虑到语料库的全局特征，直观上来说比Word2Vec更合理。
- GloVe中的很多推导都是intuitive的，实际使用中，GloVe还是没有Word2vec来的广泛。

## 第二代技术预训练技术: Contextual Embeddings
通过预训练得到高质量的词向量一直是具有挑战性的问题，主要有两方面的难点，一个是词本身具有的**语法语义复杂**属性，另一个是这些语法语义的复杂属性如何随着上下文语境产生变化，也就是**一词多义性**问题。传统的词向量方法例如word2vec、GloVe等都是训练完之后，每个词向量就固定下来，这样就无法解决一词多义的问题。接下来的模型就是基于解决这个问题展开的。

### ELMo
ELMo（Embeddings from Language Models）是有AI2提出，该模型不仅去学习**单词特征**，还有**句法特征**与**语义特征**。其通过在大型语料上预训练一个深度BiLSTM语言模型网络来获取词向量，也就是每次输入一句话，可以根据这句话的上下文语境获得每个词的向量，这样子就可以解决一词多义问题。

![ELMo](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111150545462.png)

Elmo模型的**本质思想**是先用语言模型学习一个单词的 Word Embedding，此时无法区分一词多义问题。在实际使用Word Embedding的时候，单词已经具备特定的上下文，这时可以根据上下文单词的语义调整单词的 Word Embedding 表示，这样经过调整后的 Word Embedding 更能表达上下文信息，自然就解决了多义词问题。

**评价**
- 在模型层面解决了一词多义的问题，最终得到的词向量能够随着上下文变化而变化。
- LSTM抽取特征的能力远弱于Transformer
- 拼接方式双向融合特征融合能力偏弱。

### GPT
GPT（Generative Pre-Training）模型用单向Transformer代替ELMo的LSTM来完成预训练任务，其将12个Transformer叠加起来。训练的过程较简单，将句子的n个词向量加上位置编码(positional encoding)后输入到 Transformer中 ，n个输出分别预测该位置的下一个词。

 GPT的单向Transformer结构和GPT的模型结构，如图所示：
![GPT](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111150547484.png)

**评价**
- 第一个结合 Transformer 架构（Decoder）和自监督预训练目标的模型
- 语言模型使用的是单行语言模型为目标任务。

### BERT
BERT采用和GPT完全相同的两阶段模型，首先是语言模型预训练，其次是后续任务的拟合训练。和GPT最主要不同在于预训练阶段采了类似ELMo的双向语言模型技术、MLM(mask language model)技术以及 NSP(next sentence prediction) 机制。

![BERT](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111150548149.png)

**评价**
- 采用了Transformer结构能够更好的捕捉全局信息。
- 采用双向语言模型，能够更好的利用了上下文的双向信息。
- mask不适用于自编码模型，[Mask]的标记在训练阶段引入，但是微调阶段看不到。


## 延伸方向
### 研究方向
预训练模型延伸出了很多新的研究方向。包括了：
- 基于知识增强的预训练模型，Knowledge-enriched PTMs
- 跨语言或语言特定的预训练模型，multilingual or language-specific PTMs
- 多模态预训练模型，multi-modal PTMs
- 领域特定的预训练模型，domain-specific PTMs
- 压缩预训练模型，compressed PTMs
![预训练的延伸方向](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111150915051.png)
摘自《Pre-trained models for natural language processing: A survey》

### 模型衍生
![模型衍生](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111160553063.png)
摘自《Pre-Trained Models: Past, Present and Future》


## 应用于下游任务
### 迁移学习
![迁移学习](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202111150901877.png)

- 不同的PTMs在相同的下游任务上有着不同的效果，这是因为PTMs有着不同的预训练任务，模型架构和语料。针对不同的下游任务需要**选择合适的预训练任务、模型架构和语料库**。
- 给定一个预训练的模型，不同的网络层捕获了不同的信息，基础的句法信息出现在浅层的网络中，高级的语义信息出现在更高的层级中。针对不通的任务需要**选择合适的网络层**。
- 主要有两种方式进行模型迁移：**特征提取**（预训练模型的参数是固定的）和**模型微调**（预训练模型的参数是经过微调的）。当采用特征提取时，预训练模型可以被看作是一个特征提取器，但以特征提取的方式需要更复杂的特定任务的架构。除此之外，我们应该采用内部层作为特征，因为他们通常是最适合迁移的特征。所以**微调是一种更加通用和方便的处理下游任务的方式**。

### 微调策略
微调的过程通常是比较不好预估的，即使采用相同的超参数，不同的随机数种子也可能导致差异较大的结果。除了标准的微调外，如下为一些有用的微调策略：
- 两步骤微调：两阶段的迁移，在预训练和微调之间引入了一个中间阶段。在第一个阶段，PTM 通过一个中间任务或语料转换为一个微调后的模型，在第二个阶段，再利用目标任务进行微调。
- 多任务微调：在多任务学习框架下对其进行微调。
- 利用额外模块进行微调：微调的主要缺点就是其参数的低效性。每个下游模型都有其自己微调好的参数，因此一个更好的解决方案是将一些微调好的适配模块注入到PTMs中，同时固定原始参数。

## 参考资料
- [复旦大学最新《预训练语言模型》2020综述论文大全](https://mp.weixin.qq.com/s/kwKZfNSYTzc-PGKxTxm8-w)
- [面向自然语言处理的预训练技术研究综述/李舟军](http://www.jsjkx.com/CN/article/openArticlePDF.jsp?id=18933)
- [请问深度学习中预训练模型是指什么？如何得到？/ 微软亚洲研究院的回答 / 知乎](https://www.zhihu.com/question/327642286/answer/1465037757)
- [A Neural Probabilistic Language Model/ Bengio](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [A Neural Probabilistic Language Model/ paperweekly/ zhihu](https://zhuanlan.zhihu.com/p/21240807)
- [Efficient Estimation of Word Representations in Vector Space / Tomas Mikolov](https://arxiv.org/pdf/1301.3781.pdf)
- [GloVe: Global Vectors for Word Representation / Jeffrey Pennington](https://nlp.stanford.edu/pubs/glove.pdf)
- [Attention Is All You Need -- Transformer / zhanghao](https://imzhanghao.com/2021/09/18/transformer/)
- [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf)
- [Pre-trained models for natural language processing: A survey](https://arxiv.org/pdf/2003.08271.pdf)
- [Pre-Trained Models: Past, Present and Future](https://arxiv.org/pdf/2106.07139.pdf)
