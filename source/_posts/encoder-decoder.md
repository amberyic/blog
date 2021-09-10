---
title: 基于Encoder-Decoder框架实现Seq2Seq模型
date: 2021-08-26
categories:
- 机器学习
tags:
- 编码器-解码器
- Seq2Seq
description:
- 分别介绍了Encoder-Decoder框架和Seq2Seq模型的结构，完成的工作，以及两者之间的关系。最后给予Encoder-Decoder的框架实现了Seq2Seq的模型，方便大家理解。
---

## Encoder-Decoder简介
Encoder-Decoder框架是一种文本处理领域的研究模式，他并不是特指某种具体的算法，而是一类算法统称。Encoder和Decoder部分可以是任意的文字，语音，图像，视频数据，模型可以采用CNN，RNN，BiRNN、LSTM、GRU等等。所以基于Encoder-Decoder，我们可以设计出各种各样的应用算法。

![Encoder-Decoder框架](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/20210526143504.png)

**应用场景**
- 文字-文字：机器翻译，对话机器人，文章摘要，代码补全
- 音频-文字：语音识别
- 图片-文字：图像描述生成

## Encoder-Decoder结构
Cho在2014年提出了[Encoder–Decoder结构](https://arxiv.org/pdf/1406.1078.pdf)，它由两个RNN组成，另外本文还提出了GRU的门结构，相比LSTM更加简洁，而且效果不输LSTM。
![RNN Encoder–Decoder](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/20210827085803.png)

Encoder-Decoder将可变长度序列编码为固定长度向量，然后将定长度向量表示解码回可变长度序列。可以形式化为：$p\left(y_{1}, \ldots, y_{T^{\prime}} \mid x_{1}, \ldots, x_{T}\right)$，这里$T$和$T^{\prime}$可以不一样，即输入的长度跟输出的长度可以不一致。

**Encoder**是一个RNN，他顺序地读取输入序列$x$的每个符号，当读到一个符号时，RNN的隐藏状态$h$会根据下面的等式发生变化。在读取序列的结尾（用序列结束符号标记）后，RNN的隐藏状态是整个输入序列的摘要$c$。
$$\mathbf{h}_{\langle t\rangle}=f\left(\mathbf{h}_{\langle t-1\rangle}, x_{t}\right)$$
> $x$是输入序列 $\mathbf{x}=\left(x_{1}, \ldots, x_{T}\right)$
> $f$是非线性激活函数。$f$可能像逻辑回归sigmoid函数一样简单，也可能像LSTM单元一样复杂

**Decoder**是另一个RNN，他被用来生成输出序列，根据Encoder生成的摘要$c$和后续隐状态和输入状态来得到后续状态，Decoder中t时刻内部状态的$h_t$为：
$$\mathbf{h}_{\langle t\rangle}=f\left(\mathbf{h}_{\langle t-1\rangle}, y_{t-1}, \mathbf{c}\right)$$
该时刻的概率表示为：
$$P\left(y_{t} \mid y_{t-1}, y_{t-2}, \ldots, y_{1}, \mathbf{c}\right)=g\left(\mathbf{h}_{\langle t\rangle}, y_{t-1}, \mathbf{c}\right)$$

Encoder和Decoder这两个模块联合训练去最大化给定输入序列$x$时输出序列为$y$的条件概率:
$$\max _{\boldsymbol{\theta}} \frac{1}{N} \sum_{n=1}^{N} \log p_{\boldsymbol{\theta}}\left(\mathbf{y}_{n} \mid \mathbf{x}_{n}\right)$$
> $θ$ 是一系列的模型参数
> $(x_n, y_n)$ 是训练集中(输入序列, 输出序列)的一组样本

## Seq2Seq
Seq2Seq（是 Sequence-to-sequence 的缩写），他输入一个序列，输出另一个序列。这种结构最重要的地方在于输入序列和输出序列的长度是可变的。

2014年Google的Sutskever提出了[Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf)，只不过比Cho晚了一点。论文中的模型结构更简单，Decoder在t时刻yt是由ht，yt−1决定，而没有c，Encoder 和 Decoder都用的LSTM结构。
![Seq2Seq](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/20210827093601.png)

## Encoder-Decoder和Seq2Seq的关系
这两种叫法基本都是前后脚被提出来的，其实是技术发展到一定阶段自然的一次演进，基本上可以划上等号，如果非要讲他们的差别，那么就只能说下面着两条了。
- Seq2Seq使用的具体方法基本都属于Encoder-Decoder模型的范畴。
- Seq2Seq不特指具体方法，只要满足输入序列到输出序列的目的，都可以统称为Seq2Seq模型，即Seq2Seq强调目的，Encoder-Decoder强调方法。


## 代码实现
下面是一个Seq2Seq模型在机器翻译中使用的示意图。编码器位于左侧，仅需要源语言的序列作为输入。解码器位于右边，需要两种版本的目标语言序列，一种用于输入，一种用于目标（Loss计算）
![Seq2Seq模型在机器翻译中使用的示意图](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/202108280938271.png)

网上找到了一个比较好的[实现](https://github.com/ChunML/NLP/blob/master/machine_translation/train_simple_tf2.py)，基于Tensorflow2.x的KerasAPI实现，可读性很高。

**模型定义**
模型结构定义部分，Encoder和Docoder都是继承tf.keras.Model基类构建自定义模型，实现了__init__和call方法。
- vocab_size： 训练数据词表大小
- embedding_size：词嵌入的维度，一般越大计算成本越高，建议<10
- lstm_size：LSTM的输出维度

``` python
# Encoder的实现
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
​
    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
​
        return output, state_h, state_c
​
    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))
```

``` python
# Decoder的实现
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)

        return logits, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))
```

**模型训练**
模型训练部分，使用自定义网络循环的方式进行训练。
- 因为是分类问题，所以我们选CrossEntropy作为损失函数。
- 把原始的序列输入到Encoder中，得到encoder hidden state
- 将encoder hidden state和decode input输入到Decoder，Decoder的decode input以<SOS>(Start of Sentence)为开始
- 然后计算损失，计算梯度，更新模型参数。
``` python
# 损失函数
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_states = en_states

        de_outputs = decoder(target_seq_in, de_states)
        logits = de_outputs[0]
        loss = loss_func(target_seq_out, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss

NUM_EPOCHS = 300
for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)
    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))
```

## 参考文献
[1][一文看懂 NLP 里的模型框架 Encoder-Decoder 和 Seq2Seq / easyAI](https://easyaitech.medium.com/%E4%B8%80%E6%96%87%E7%9C%8B%E6%87%82-nlp-%E9%87%8C%E7%9A%84%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6-encoder-decoder-%E5%92%8C-seq2seq-1012abf88572)
[2][Re:从零开始的机器学习 – Encoder-Decoder架构](https://flashgene.com/archives/38604.html)
[3][Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
[4][Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
[5][seq2seq 入门/简书/不会停的蜗牛](https://www.jianshu.com/p/1d3de928f40c)
[6][Sequence-to-Sequence Models: Encoder-Decoder using Tensorflow 2](https://towardsdatascience.com/sequence-to-sequence-models-from-rnn-to-transformers-e24097069639)
[7][Neural Machine Translation With Attention Mechanism](https://trungtran.io/2019/03/29/neural-machine-translation-with-attention-mechanism/)
[8][Tensorflow 2.0 之“机器翻译”](https://zhuanlan.zhihu.com/p/61509099)
[9][ChunML/NLP/machine_translation](https://github.com/ChunML/NLP/tree/master/machine_translation)
