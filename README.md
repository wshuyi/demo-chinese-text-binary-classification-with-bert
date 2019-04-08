# 如何用 Python 和 BERT 做中文文本二元分类？

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-38-00-008905.png)

# 兴奋

去年， Google 的 BERT 模型一发布出来，我就很兴奋。

因为我当时正在用 fast.ai 的 ULMfit 做自然语言分类任务（还专门写了《[如何用 Python 和深度迁移学习做文本分类？](https://zhuanlan.zhihu.com/p/48182945)》一文分享给你）。ULMfit 和 BERT 都属于预训练语言模型（Pre-trained Language Modeling），具有很多的相似性。

所谓语言模型，就是利用深度神经网络结构，在海量语言文本上训练，以抓住一种语言的**通用特征**。

上述工作，往往只有大机构才能完成。因为**花费**实在太大了。

这花费包括但不限于：

- 存数据
- 买（甚至开发）运算设备
- 训练模型（以天甚至月计）
- 聘用专业人员
- ……

**预训练**就是指他们训练好之后，把这种结果开放出来。我们普通人或者小型机构，也可以**借用**其结果，在自己的专门领域文本数据上进行**微调**，以便让模型对于这个专门领域的文本有非常清晰的**认识**。

所谓认识，主要是指你遮挡上某些词汇，模型可以较准确地猜出来你藏住了什么。

甚至，你把两句话放在一起，模型可以判断它俩是不是紧密相连的上下文关系。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-38-34-983938.png)

这种“认识”有用吗？

当然有。

BERT 在多项自然语言任务上测试，不少结果已经超越了人类选手。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-38-58-744810.png)

BERT 可以辅助解决的任务，当然也包括文本分类（classification），例如情感分类等。这也是我目前研究的问题。

# 痛点

然而，为了能用上 BERT ，我等了很久。

Google 官方代码早已开放。就连 Pytorch 上的实现，也已经迭代了多少个轮次了。

但是我只要一打开他们提供的样例，就头晕。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928502.jpeg)

单单是那代码的行数，就非常吓人。

而且，一堆的数据处理流程（Data Processor） ，都用数据集名称命名。我的数据不属于上述任何一个，那么我该用哪个？

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345044.jpeg)

还有莫名其妙的无数旗标（flags） ，看了也让人头疼不已。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345033.jpeg)

让我们来对比一下，同样是做分类任务，Scikit-learn 里面的语法结构是什么样的。

```python
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
```

即便是图像分类这种数据吞吐量大，需要许多步骤的任务，你用 fast.ai ，也能几行代码，就轻轻松松搞定。

```python
!git clone https://github.com/wshuyi/demo-image-classification-fastai.git
from fastai.vision import *
path = Path("demo-image-classification-fastai/imgs/")
data = ImageDataBunch.from_folder(path, test='test', size=224)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(8, 8))
```

别小瞧这几行代码，不仅帮你训练好一个图像分类器，还能告诉你，那些分类误差最高的图像中，模型到底在关注哪里。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928501.png)

对比一下，你觉得 BERT 样例和 fast.ai 的样例区别在哪儿？

我觉得，后者是**给人用的**。

# 教程

我总以为，会有人把代码重构一下，写一个简明的教程。

毕竟，文本分类任务是个常见的机器学习应用。应用场景多，也适合新手学习。

但是，这样的教程，我就是没等来。

当然，这期间，我也看过很多人写的应用和教程。

有的就做到把一段自然语言文本，转换到 BERT 编码。戛然而止。

有的倒是认真介绍怎么在官方提供的数据集上，对 BERT 进行“稍微修改”使用。所有的修改，都在原始的 Python 脚本上完成。那些根本没用到的函数和参数，全部被保留。至于别人如何复用到自己的数据集上？人家根本没提这事儿。

我不是没想过从头啃一遍代码。想当年读研的时候，我也通读过仿真平台上 TCP 和 IP 层的全部 C 代码。我确定眼前的任务，难度更低一些。

但是我真的懒得做。我觉得自己被 Python 机器学习框架，特别是 fast.ai 和 Scikit-learn 宠坏了。

后来， Google 的开发人员把 BERT 弄到了 Tensorflow Hub 上。还专门写了个 Google Colab Notebook 样例。

看到这个消息，我高兴坏了。

我尝试过 Tensorflow Hub 上的不少其他模型。使用起来很方便。而 Google Colab 我已在《[如何用 Google Colab 练 Python？](https://zhuanlan.zhihu.com/p/57100935)》一文中介绍给你，是非常好的 Python 深度学习练习和演示环境。满以为双剑合璧，这次可以几行代码搞定自己的任务了。

且慢。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345041.png)

真正打开一看，还是以样例数据为中心。

普通用户需要什么？需要一个接口。

你告诉我输入的标准规范，然后告诉我结果都能有什么。即插即用，完事儿走人。

一个文本分类任务，原本不就是给你个训练集和测试集，告诉你训练几轮练多快，然后你告诉我准确率等结果吗？

你至于让我为了这么简单的一个任务，去读几百行代码，自己找该在哪里改吗？

好在，有了这个样例做基础，总比没有好。

我耐下心来，把它整理了一番。

声明一下，我并没有对原始代码进行大幅修改。

所以不讲清楚的话，就有剽窃嫌疑，也会被鄙视的。

这种整理，对于会 Python 的人来说，没有任何技术难度。

可正因为如此，我才生气。这事儿难做吗？Google 的 BERT 样例编写者怎么就不肯做？

从 Tensorflow 1.0 到 2.0，为什么变动会这么大？不就是因为 2.0 才是给人用的吗？

你不肯把界面做得清爽简单，你的竞争者（TuriCreate 和 fast.ai）会做，而且做得非常好。实在坐不住了，才肯降尊纡贵，给普通人开发一个好用的界面。

教训啊！为什么就不肯吸取呢？

我给你提供一个 Google Colab 笔记本样例，你可以轻易地替换上自己的数据集来运行。你需要去理解（包括修改）的代码，**不超过10行**。

我先是测试了一个英文文本分类任务，效果很好。于是写了[一篇 Medium 博客](https://towardsdatascience.com/how-to-do-text-binary-classification-with-bert-f1348a25d905)，旋即被 Towards Data Science 专栏收录了。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345050.jpeg)

Towards Data Science 专栏编辑给我私信，说：

> Very interesting, I like this considering the default implementation is not very developer friendly for sure.

有一个读者，居然连续给这篇文章点了50个赞（Claps），我都看呆了。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345040.png)

看来，这种忍受已久的痛点，不止属于我一个人。

估计你的研究中，中文分类任务可能遇到得更多。所以我干脆又做了一个中文文本分类样例，并且写下这篇教程，一并分享给你。

咱们开始吧。


# 代码

请点击[这个链接](https://github.com/wshuyi/demo-chinese-text-binary-classification-with-bert/blob/master/demo_chinese_text_binary_classification_with_bert.ipynb)，查看我在 Github 上为你做好的 IPython Notebook 文件。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345039.jpeg)


Notebook 顶端，有个非常明显的 "Open in Colab" 按钮。点击它，Google Colab 就会自动开启，并且载入这个 Notebook 。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345030.png)

我建议你点一下上图中红色圈出的 “COPY TO DRIVE” 按钮。这样就可以先把它在你自己的 Google Drive 中存好，以便使用和回顾。

这件事做好以后，你实际上只需要执行下面三个步骤：

1. 你的数据，应该以 Pandas 数据框形式组织。如果你对 Pandas 不熟悉，可以参考我的[这篇文章](https://www.jianshu.com/p/a7a7db17e26d)。
2. 如有必要，可以调整训练参数。其实主要是训练速率(Learning Rate)和训练轮数(Epochs)。
3. 执行 Notebook 的代码，获取结果。

当你把 Notebook 存好之后。定睛一看，或许会觉得上当了。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345053.png)

> 老师你骗人！说好了不超过10行代码的！

**别急**。

在下面这张图红色圈出的这句话之前，你**不用修改任何内容**。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928507.png)

请你点击这句话所在位置，然后从菜单中如下图选择 `Run before` 。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928498.png)

下面才都是紧要的环节，集中注意力。

第一步，就是把数据准备好。

```python
!wget https://github.com/wshuyi/demo-chinese-text-binary-classification-with-bert/raw/master/dianping_train_test.pickle

with open("dianping_train_test.pickle", 'rb') as f:
    train, test = pickle.load(f)
```

这里使用的数据，你应该并不陌生。它是餐饮点评情感标注数据，我在《[如何用Python和机器学习训练中文文本情感分类模型？](https://zhuanlan.zhihu.com/p/34482959)》和《[如何用 Python 和循环神经网络做中文文本分类？](https://zhuanlan.zhihu.com/p/50488163)》中使用过它。只不过，为了演示的方便，这次我把它输出为 pickle 格式，一起放在了演示 Github repo 里，便于你下载和使用。

其中的训练集，包含1600条数据；测试集包含400条数据。标注里面1代表正向情感，0代表负向情感。

利用下面这条语句，我们把训练集重新洗牌(shuffling)，打乱顺序。以避免过拟合(overfitting)。

```python
train = train.sample(len(train))
```

这时再来看看我们训练集的头部内容。

```python
train.head()
```

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345042.png)

如果你后面要替换上自己的数据集，请注意格式。训练集和测试集的列名称应该保持一致。

第二步，我们来设置参数。

```python
myparam = {
        "DATA_COLUMN": "comment",
        "LABEL_COLUMN": "sentiment",
        "LEARNING_RATE": 2e-5,
        "NUM_TRAIN_EPOCHS":3,
        "bert_model_hub":"https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
    }
```

前两行，是把文本、标记对应的列名，指示清楚。

第三行，指定训练速率。你可以阅读原始论文，来进行超参数调整尝试。或者，你干脆保持默认值不变就可以。

第四行，指定训练轮数。把所有数据跑完，算作一轮。这里使用3轮。

最后一行，是说明你要用的 BERT 预训练模型。咱们要做中文文本分类，所以使用的是这个中文预训练模型地址。如果你希望用英文的，可以参考[我的 Medium 博客文章](https://towardsdatascience.com/how-to-do-text-binary-classification-with-bert-f1348a25d905)以及对应的[英文样例代码](https://github.com/wshuyi/demo_text_binary_classification_bert/blob/master/demo_text_binary_classification_with_bert.ipynb)。

最后一步，我们依次执行代码就好了。

```python
result, estimator = run_on_dfs(train, test, **myparam)
```

注意，执行这一句，可能需要**花费一段时间**。做好心理准备。这跟你的数据量和训练轮数设置有关。

在这个过程中，你可以看到，程序首先帮助你把原先的中文文本，变成了 BERT 可以理解的输入数据格式。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345037.png)

当你看到下图中红色圈出文字时，就意味着训练过程终于结束了。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345057.png)

然后你就可以把测试的结果打印出来了。

```python
pretty_print(result)
```

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-07-16-37-22-345046.png)

跟咱们之前的[教程](https://zhuanlan.zhihu.com/p/50488163)（使用同一数据集）对比一下。

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2018-11-21-08-47-47-758482.png)

当时自己得写那么多行代码，而且需要跑10个轮次，可结果依然没有超过 80% 。这次，虽然只训练了3个轮次，但准确率已经超过了 88% 。

在这样小规模数据集上，达到这样的准确度，不容易。

BERT **性能之强悍**，可见一斑。

# 小结

讲到这里，你已经学会了如何用 BERT 来做中文文本二元分类任务了。希望你会跟我一样开心。

如果你是个资深 Python 爱好者，请帮我个忙。

还记得这条线之前的代码吗？

![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-04-05-16-49-01-928506.png)

能否帮我把它们打个包？这样咱们的演示代码就可以更加短小精悍和清晰易用了。

欢迎在[咱们的 Github 项目](https://github.com/wshuyi/demo-chinese-text-binary-classification-with-bert)上提交你的代码。如果你觉得这篇教程对你有帮助，欢迎给[这个 Github 项目](https://github.com/wshuyi/demo-chinese-text-binary-classification-with-bert)加颗星。谢谢！

祝深度学习愉快！

# 延伸阅读

你可能也会对以下话题感兴趣。点击链接就可以查看。

- [如何高效学 Python ？](https://zhuanlan.zhihu.com/p/29631043)
- [学 Python ，能提升你的竞争力吗？](https://zhuanlan.zhihu.com/p/53011746)
- [文科生如何理解卷积神经网络？](https://zhuanlan.zhihu.com/p/36416075)
- [文科生如何理解循环神经网络（RNN）？](https://zhuanlan.zhihu.com/p/49988171)
- [《文科生数据科学上手指南》分享](https://zhuanlan.zhihu.com/p/44653452)

喜欢请点赞和打赏。还可以微信关注和置顶我的公众号[“玉树芝兰”(nkwangshuyi)](https://i.loli.net/2019/03/05/5c7dd41f11372.png)。

如果你对 Python 与数据科学感兴趣，不妨阅读我的系列教程索引贴《[如何高效入门数据科学？](https://zhuanlan.zhihu.com/p/35563090)》，里面还有更多的有趣问题及解法。

知识星球入口在这里：

![](https://i.loli.net/2019/03/05/5c7dd41f11372.png)
