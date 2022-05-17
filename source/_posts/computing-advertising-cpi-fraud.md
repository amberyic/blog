---
title: CPI/CPA广告常见作弊方法总结
date: 2021-11-01
updated: 2021-11-21
categories:
- 计算广告
tags:
- 广告归因
- 反作弊
- 计算广告
keywords: 计算广告,反作弊,广告归因,伪造归因,点击欺诈,点击劫持,伪造用户,模拟器,设备农场,SDK伪造
description: 介绍了应用类广告第三方归因的方法，详细分析了其中的作弊类型以及各种类型的具体作弊方法。
---

## 广告归因方式
要讨论广告作弊，就需要先了解广告的归因逻辑，作弊手段基本都是围绕广告归因的逻辑来进行的。
广告归因方案多种多样，我们这里主要讨论**应用类广告、海外移动生态、第三方归因**的方案。
海外移动广告生态，拥有比较成熟可信的第三方归因平台，比如Appflyer，Adjust以及Kochava等等。归因的核心逻辑是最后归因模型，即“Last Click”。

![应用广告归因](https://oss.imzhanghao.com/img/202110271433410.png)
媒体的广告曝光后，若用户对广告进行了点击，媒体会将广告点击的媒体信息、用户设备信息（核心是IDFA/IMEI）、时间戳、网络状态等信息通过302跳转的方式给到第三方归因平台（即广告点击后，会通过302重定向跳转到第三方归因平台的后台，然后再跳到Google Play或者App Store）。此时，第三方归因平台其实是没有广告的曝光相关信息的。

应用激活后，可以通过接入归因SDK或者通过服务端对接的方式（S2S）的方式将应用相关信息回传给第三方归因平台，归因平台从数据库中找出匹配的媒体点击信息，通过匹配的应用包名、用户ID信息和广告的点击信息，按照最后一次点击的逻辑将应用的激活归因给对应的媒体和广告，完成一次归因的流程。

## 作弊的分类
在全部广告作弊类型中，作弊者能够伪造在归因中使用的任ㄧ或两类“信号”（Signal）。这两类信号分别为广告交互（例如查看或点击，对应归因方式中2的位置）和应用活动（例如安装、会话或事件，对应归因方式中3的位置）。在此基础上，我们将作弊分为伪造广告交互和伪造用户应用内活动。前者称为**伪造归因（Spoofed Attribution）**，后者被称为**伪造用户（Spoofed Users）**。
![作弊分类](https://oss.imzhanghao.com/img/202110271118560.png)
- 类型1中的全部流量均为真实流量，即用户受到广告驱使，与应用所产生的真实互动。
- 类型2指伪造归因，即作弊者伪造真实用户的广告交互。其目的是为了窃取用户与应用之间的自然交互或通过真实广告所产生的效果。此类型的伪造也被称为“窃取归因”或“流量盗取”(poaching)。
- 类型3和4指伪造用户。此作弊类型专注于模拟用户的应用内活动行为。通过伪造不存在的用户而产生的应用安装和事件，作弊者可以窃取以应用转化为奖励的广告预算。“外挂”、“虚拟机器人”以及任何与“虚假用户”相关的手段都能归纳为此类型的作弊。

## 伪造归因
伪造归因，也称Attribution Fraud、Spoofed Attribution、归因作弊、抢归因，是利用归因逻辑上的一些漏洞进行作弊的手段，通过发布虚假的曝光/点击，劫持真实用户产生的转化。

### 点击欺诈(Click Spamming)
点击欺诈，也叫Click Stuffing或Click Flood，中文名叫点击泛滥、点击填塞、大点击、预点击、撞库，是伪造海量广告的曝光或点击，等到用户真安装之后，在Last Click归因原则下，如点击后N天内安装的都算成带来点击的渠道，将其他渠道或者是自然量归因抢到自己的渠道中来。

欺诈性应用程序可能会在用户使用它时执行点击，或者在后台活动（例如启动、省电等）时执行点击。该应用程序甚至可以将展示次数报告为点击以呈现虚假的广告交互，而这一切都是在用户无意或不知情的情况下进行的。

![大点击](https://oss.imzhanghao.com/img/202110271647119.png)

**点击欺诈的形式**：
- 广告堆叠点击(Ad Stacking Clicks)： 在单个广告展示位置中以层叠的方式放置多个广告，只有顶部广告可见。堆栈中的所有广告都按空间的每次展示或点击计费。欺诈者将多个广告投放到程序化广告活动中，并为未查看的广告创造收入。应用悄悄在后台加载和点击广告。
![广告堆叠点击](https://oss.imzhanghao.com/img/202110271546938.png)
- 浏览点击（Views as Clicks）或“预缓存”：以点击方式发送视图，在广告显示之前点击它们。将展示作为点击发送的渠道。
- 服务器到服务器的点击(Server2Server Clicks)：从Adx处获得流量直接给三方发送点击事件。
这些形式都具有一个相同的特征：用户实际上并没有打算与广告进行互动，也没有兴趣下载显示的应用程序。发送人工点击目录的服务器。

**依赖条件**
- 丰富的广告资源，因为点击欺诈主要是盗取自然流量，所以需要一些自然下载量比较大的应用广告资源。
- 海量的设备和流量，找到活跃的设备。

**识别方法**
- long CTIT(Click-to-install-time) distribution rates
- low click-to-install conversion rates
- high multi-touch contributor rates (or)
![CTIT](https://oss.imzhanghao.com/img/202110271900778.png)

### 点击劫持(Click Injection)
点击劫持也叫Install Hijacking、点击注入、小点击，指的是作弊者通过安装在用户设备上的一个应用程序来“监听”其他应用程序的安装广播消息。当用户设备上安装了新的应用程序时，作弊者就会收到通知，然后在安装完成之前发送虚假点击利用归因模型的漏洞劫取相应的安装。特点是点击到安装时间过短，应用商店记录的下载时间早于点击广告的时间。
![小点击](https://oss.imzhanghao.com/img/202110271649228.png)

如果我们知道一个应用的下载或者安装时间点，在这个时候将“点击”信息发送给第三方归因平台，由于这个时候离应用的激活更近，按照Last Click原理归因概率就非常高。而Android系统刚好提供了获取应用安装的广播机制。当应用安装的时候，Android系统会将应用安装的消息（android.intent.action.PACKAGE_ADDED）通过系统广播（Broadcast）的方式广播给在已经在系统注册文件上（Manifest.xml）注册了安装广播监听能力的应用。获取到应用的安装信息（核心信息是应用的包名）之后，此时广告联盟SDK就会根据这个包名从广告后台中获取对应的广告信息，并将相关的用户设备信息，媒体信息通过“虚拟点击”的方式传到第三方归因平台。

**依赖条件**
- 丰富的广告资源，因为广告信息是在收到系统应用安装广播之后，实时根据包名从广告后台请求拉取，然后才做的“模拟点击”信息发送。否则的话，你都不知道要给第三方归因平台发送什么样的广告点击信息。
- 注册系统的应用安装广告广播能力（或者知道Google play的下载事件）。这样才能知道应用什么时候被安装。同时联盟SDK的流量覆盖面要广，这样就可以抢到更多的广告。这个现象白热化的时候，有些小的广告联盟甚至只需要流量媒体接入他们的SDK而无需展示广告就可以获取收入。

**识别方法**
- short CTIT(Click-to-install-time) distribution rates
- high click-to-install conversion rates
![CTIT](https://oss.imzhanghao.com/img/202110271901841.png)

根据安装的不同来源，我们的过滤方法稍有差异。
- Google paly和华为:Google 和华为的 referrer API 会创建时间戳，这些时间戳可以用来甄别是否出现了点击劫持。首先，我们会将点击的时间与 intall_begin_time 做比对；如果点击发生在该时间戳后，基本可以肯定就是点击劫持。SDK还会收集install_finish_time时间戳，进行第二层过滤。
- 其他渠道的安装​:发生在 Google Play 应用商店和华为 AppGallery 之外的安装没有 referrer API，无法发送 install_begin 时间戳。因此，要过滤此类安装，我们要依赖于 install_finish_time 时间戳。 install_finish_time 时间戳后接收到的点击将被视为欺诈并被拒绝。

## 伪造用户
伪造用户发生虚假应用内活动，我们能够发现模拟器、设备农场(Device Farms) 和 SDK 伪造。
在最初发现的伪造用户案例中，我们检测到欺诈者利用模拟器模仿云计算服务上真实用户使用安卓应用的情况。同时，我们还识别出东南亚国家的iOS设备农场，他们通过真实的设备和人员伪造了虚假的应用活动。

### 模拟器(Bots)
模拟器指的是作弊者通过自动化脚本或计算机程序模拟真实用户的点击、下载、安装甚至是应用内行为，伪装成为真实用户， 从而骗取广告主的CPI/CPA预算。
![模拟器](https://oss.imzhanghao.com/img/202110271700101.png)
**特点**是IP离散度密集、新设备率过高、用户行为异常、机型/系统/时间等分布异常等。

### 设备农场(Device Farms)
设备农场指的是作弊者购买大量真实设备进行广告点击、下载、安装和应用内行为，并通过修改设备广告跟踪符等方式隐藏设备信息。
![设备农场](https://oss.imzhanghao.com/img/202110271913495.png)

设备农场主使用各种策略来隐藏他们的活动，包括隐藏在新的IP地址后面，使用各种设备，同时启用限制广告跟踪或隐藏在 DeviceID重置欺诈后面（每次安装时重置他们的 DeviceID）。当大规模实施时，这种欺诈也称为DeviceID重置Marathons。
![设备农场操作流程](https://oss.imzhanghao.com/img/202110271910424.png)
**特点**是IP离散度密集、新设备率过高、用户行为异常、机型/系统分布异常等

### SDK伪造(SDK Spoofing)
SDK伪造是指作弊者通过执行“中间人攻击”破解第三方SDK的通信协议后，在没有任何实际安装的情况下，使用真实设备的数据来发送虚假的点击和安装，以此消耗广告主的预算的作弊行为。作弊者毁坏加密和哈希签名，进而引发了作弊者和研究人员之间的对决。

**特点**是广告主后台数据和第三方数据不符。


## 反作弊方法
### 匿名IP
匿名IP过滤器可保护应用跟踪数据的真实性，使其免受来自VPN、Tor出口节点或数据中心的欺诈安装活动影响。一些欺诈者使用模拟软件伪造安装，并将欺诈转化放到高价值市场获取利润，匿名IP过滤器针对的就是这些欺诈者。

### 点击安装时间
点击安装时间(Click to install time,CTIT)衡量用户旅程中时间戳之间的伽玛分布 - 用户的初始广告互动和他们的首次应用启动。

![CTIT](https://oss.imzhanghao.com/img/202111201343440.png)

CTIT 可用于识别基于点击的欺诈的不同案例:
- 短 CTIT（低于 10 秒）：可能存在安装劫持欺诈(install hijacking)
- 长时间 CTIT（24 小时及之后）：可能的大点击欺诈(click flooding)

### 新设备率
新设备率(New device rate, NDR)将突出显示下载广告商应用的新设备的百分比。

有新设备当然是正常的，因为会有新用户安装应用程序或者现有用户更换设备。但是，必须密切关注其活动可接受的新设备率，因为该比率由测量的新设备ID决定。因此，它可以被设备ID重置欺诈策略所操纵，这在设备农场中很常见。

![NDR](https://oss.imzhanghao.com/img/202111201721503.png)

### 传感器
设备传感器(Device sensors)可以收集设备电池电量到倾斜角度等上百个指标，可以用来进行生物识别行为分析。

![传感器](https://oss.imzhanghao.com/img/202111201354087.png)

这些指标有助于为每次安装创建配置文件——分析每次安装的设备和用户行为及其与真实用户测量的正常趋势的兼容性。

### 限制广告跟踪
限制广告跟踪（Limit ad tracking， LAT）是一项隐私功能，允许用户限制广告商收到的有关其设备生成的活动的数据。当用户启用LAT时，广告商及其测量解决方案会收到一个空白设备ID，而不是特定于设备的ID。

这个指标仅在Google和iOS广告标识符相关，亚马逊、小米等使用其他标识符。

### 转化率
转化率（Conversion rates）描述了一种操作到另一种操作的转化，这可能意味着广告展示转化为点击、点击转化为安装或安装给活跃用户。 广告商在用户旅程中的任何一点了解其预期转化率有助于防止欺诈渗透。

转化率过高可能不是真的，会被怀疑有作弊嫌疑。

### 人工智能
人工智能已成为常见的欺诈指标，因为它允许大规模应用欺诈识别逻辑。人工智能有助于指示人类无法追踪的任何规模的实例。

机器学习算法（即**贝叶斯网络**）与大型移动归因数据库相结合，将确保提供高效准确的欺诈检测解决方案。


## 参考资料
- [1][《深入分析广告归因》/ PMCoder](https://toutiao.io/posts/63t2w5v/preview)
- [2][《Adjust CTO 深度剖析移动作弊: 打击作弊需从定义开始（一）》 / Paul Müller / CTO of Adjust](https://mp.weixin.qq.com/s/1V8IwO-H9E1I1odxYnk_Ww)
- [3][《What is Ad Stacking?》 / Fraud Blocker](https://fraudblocker.com/articles/what-is-ad-stacking)
- [4][广告堆栈点击 / kochava support](https://support.kochava.com/fraud-console/ad-stacking-clicks/)
- [5][《移动广告作弊现形：三组实例的探讨与解决方案》 / Paul Müller / CTO of Adjust](https://www.adjust.com/zh/blog/mobile-fraud-in-practice-three-real-world-examples-zh/)
- [6][《Mobvista 移动广告反作弊白皮书》](https://www.mobvista.com/wp-content/themes/mobvista/dist/global/files/white-book.pdf?62c0887b)
- [7][《Click flooding》 / appsflyer](https://www.appsflyer.com/glossary/click-flooding/)
- [8][《How CTIT is Used to Detect Mobile Ad Fraud》 / interceptd](https://interceptd.com/how-ctit-click-to-install-time-is-used-to-detect-mobile-ad-fraud/)
- [9][《SDK spoofing》/ appsflyer](https://www.appsflyer.com/glossary/sdk-spoofing/)
- [10][mobile-ad-fraud-for-marketers/ appsflyer](https://www.appsflyer.com/resources/guides/mobile-ad-fraud-for-marketers/)
