---
title: 五大经典算法|2.贪心算法
date: 2020-04-15
categories:
- 五大经典算法
tags:
- 贪心算法
---
> 贪心算法在思想及其实现都比较简单易懂。我们数据结构中讲解过huffman tree构建及图的最小生成树涉及贪心算法。
> 经典例题:1)活动时间安排问题,2)背包问题,3)线段覆盖(lines cover),4)数字组合问题,5)找零钱的问题,6)多机调度问题,7)小船过河问题,8)销售比赛,9)Huffman编码,10)Dijkstra算法,11)最小生成树算法
<!-- more -->

## 概念
贪心算法是指在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，只做出在某种意义上的局部最优解。贪心算法不是对所有问题都能得到整体最优解，关键是贪心策略的选择，选择的贪心策略必须具备无后效性，即某个状态以前的过程不会影响以后的状态，只与当前状态有关。

贪心算法没有固定的算法框架，算法设计的关键是贪心策略的选择。必须注意的是，贪心算法不是对所有问题都能得到整体最优解，选择的贪心策略必须具备无后效性，即某个状态以后的过程不会影响以前的状态，只与当前状态有关。


## 思想
- 1.建立数学模型来描述问题；
- 2.把求解的问题分成若干个子问题；
- 3.对每一子问题求解，得到子问题的局部最优解；
- 4.把子问题的局部最优解合成原来问题的一个解。

## 题目

### 活动时间安排问题
* 题目
> 设有N个活动时间集合，每个活动都要使用同一个资源，比如说会议场，而且同一时间内只能有一个活动使用，每个活动都有一个使用活动的开始si和结束时间fi，即他的使用区间为(si,fi),现在要求你分配活动占用时间表，即哪些活动占用该会议室，哪些不占用，使得他们不冲突，要求是尽可能多的使参加的活动最大化，即所占时间区间最大化！

> 上图为每个活动的开始和结束时间，我们的任务就是设计程序输出哪些活动可以占用会议室！

* 分析
* 代码

```
#include <iostream>
using namespace std;

void GreedyChoose(int len,int *s,int *f,bool *flag);

int main(int argc, char* argv[]) {
  int s[11] ={1,3,0,5,3,5,6,8,8,2,12};
  int f[11] ={4,5,6,7,8,9,10,11,12,13,14};

  bool mark[11] = {0};

  GreedyChoose(11,s,f,mark);
  for(int i=0;i<11;i++)
    if (mark[i])
      cout<<i<<" ";
  system("pause");
  return 0;
}

void GreedyChoose(int len,int *s,int *f,bool *flag) {
  flag[0] = true;
  int j = 0;
  for(int i=1;i<len;++i)
    if (s[i] >= f[j]) {
      flag[i] = true;
      j = i;
    }
}
```

### 背包问题
* 题目
> 有一个背包，背包容量是M=150。有7个物品，物品不可以分割成任意大小。要求尽可能让装入背包中的物品总价值最大，但不能超过总容量.

| 物品  | A | B | C | D | E |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 重量  | 35 | 30 | 60 | 50 | 40 | 10 | 25 |
| 价值  | 10 | 40 | 30 | 50 | 35 | 40 | 30 |

* 分析
> 目标函数： ∑pi最大
> 约束条件是装入的物品总重量不超过背包容量：∑wi<=M( M=150)。
> (1)根据贪心的策略，每次挑选价值最大的物品装入背包，得到的结果是否最优？
> (2)每次挑选所占重量最小的物品装入是否能得到最优解？
> (3)每次选取单位重量价值最大的物品，成为解本题的策略。
> 值得注意的是，贪心算法并不是完全不可以使用，贪心策略一旦经过证明成立后，它就是一种高效的算法。
> 贪心算法还是很常见的算法之一，这是由于它简单易行，构造贪心策略不是很困难。
> 可惜的是，它需要证明后才能真正运用到题目的算法中。
> 一般来说，贪心算法的证明围绕着：整个问题的最优解一定由在贪心策略中存在的子问题的最优解得来的。
> 对于背包问题中的3种贪心策略，都是无法成立(无法被证明)的，解释如下：
> 贪心策略：选取价值最大者。反例：
> W=30
> 物品：A B C
> 重量：28 12 12
> 价值：30 20 20
> 根据策略，首先选取物品A，接下来就无法再选取了，可是，选取B、C则更好。
> (2)贪心策略：选取重量最小。它的反例与第一种策略的反例差不多。
> (3)贪心策略：选取单位重量价值最大的物品。反例：
> W=30
> 物品：A B C
> 重量：28 20 10
> 价值：28 20 10
> 根据策略，三种物品单位重量价值一样，程序无法依据现有策略作出判断，如果选择A，则答案错误。
> 但是果在条件中加一句当遇见单位价值相同的时候,优先装重量小的,这样的问题就可以解决.
>
> 所以需要说明的是，贪心算法可以与随机化算法一起使用，具体的例子就不再多举了。
> (因为这一类算法普及性不高，而且技术含量是非常高的，需要通过一些反例确定随机的对象是什么，
> 随机程度如何，但也是不能保证完全正确，只能是极大的几率正确)。

* 代码

```
#include <iostream>
using namespace std;

struct Node {
  float weight;
  float value;
  bool mark;
  char char_mark;
  float pre_weight_value;
};

int main(int argc, char* argv[]) {
  float Weight[7] = {35,30,60,50,40,15,20};
  float Value [7] = {10,40,30,50,35,40,30};
  Node array[7];
  for(int i=0; i<7; i++) {
    array[i].value = Value[i];
    array[i].weight = Weight[i];
    array[i].char_mark = 65 + i;
    array[i].mark = false;
    array[i].pre_weight_value = Value[i] / Weight[i];
  }

  for(i=0;i<7;i++)
    cout<<array[i].pre_weight_value<<" ";
  cout<<endl;

  float weight_all=0.0;
  float value_all = 0.0;
  float max = 0.0;
  char charArray[7];
  int flag,n = 0;

  while(weight_all <= 150) {
    for(int index=0;index < 7; ++index) {
      if (array[index].pre_weight_value > max && array[index].mark == false) {
        max = array[index].pre_weight_value ;
        flag = index;
      }
    }

    charArray[n++] = array[flag].char_mark;
    array[flag].mark = true;
    weight_all += array[flag].weight;
    value_all += array[flag].value;
    max = 0.0;
  }

  for(i=0;i<n-1;i++)
    cout<<charArray[i]<<" ";
  cout<<endl;
  cout<<"weight_all:"<<weight_all- array[n-1].weight<<endl;
  cout<<"value_all:"<<value_all<<endl;

  system("pause");
  return 0;
}

```

### 线段覆盖(lines cover)
* 题目
> 在一维空间中告诉你N条线段的起始坐标与终止坐标，要求求出这些线段一共覆盖了多大的长度。
> 为了方便说明，我们采用上述表格中的数据代表10条线段的起始点和终点，注意，这里是用起始点为顺序进行排列，和上面的不一样，知道了这些我们就可以着手开始设计这个程序：


* 分析


* 代码

```
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
  int s[10] = {2,3,4,5,6,7,8,9,10,11};
  int f[10] = {3,5,7,6,9,8,12,10,13,15};
  int TotalLength = (3-2);

  for(int i=1,int j=0; i<10 ; ++i) {
    if (s[i] >= f[j]) {
      TotalLength += (f[i]-s[i]);
      j = i;
    } else {
      if (f[i] <= f[j]) {
        continue;
      } else {
        TotalLength += f[i] - f[j];
        j = i;
      }
    }
  }

  cout<<TotalLength<<endl;
  system("pause");
  return 0;
}

```


### 数字组合问题
* 题目
> 设有N个正整数，现在需要你设计一个程序，使他们连接在一起成为最大的数字，例3个整数 12,456,342 很明显是45634212为最大，4个整数 342，45,7,98显然为98745342最大
> 程序要求：输入整数N 接下来一行输入N个数字，最后一行输出最大的那个数字！


* 分析
> 拿到这题目，看起要来也简单，看起来也难，简单在什么地方，简单在好像就是寻找哪个开头最大，然后连在一起就是了，难在如果N大了，假如几千几万，好像就不是那么回事了，要解答这个题目需要选对合适的贪心策略，并不是把数字由大排到小那么简单，网上的解法是将数字转化为字符串，比如a+b和b+a，用strcmp函数比较一下就知道谁大，也就知道了谁该排在谁前面，不过我觉得这个完全没必要，在这里我采用一种比较巧妙的方法来解答，不知道大家还记得冒泡排序法不，那是排序最早接触的一种方法，我们先看看它的源代码：


* 代码

```
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
  int array[10];
  for(int i=0;i<10;i++)
    cin>>array[i];

  int temp;
  for(i=0; i<=9 ; ++i)
    for(int j=0;j<10-1-i;j++)
      if (array[j] > array[j+1]) {
        temp = array[j];
        array[j] = array[j+1];
        array[j+1] = temp;
      }
  for(i=0;i<10;i++)
    cout<<array[i]<<" ";
  cout<<endl;
  system("pause");
  return 0;
}
```


### 找零钱的问题
* 题目
> 在贪心算法里面最常见的莫过于找零钱的问题了，题目大意如下，对于人民币的面值有1元 5元 10元 20元 50元 100元，下面要求设计一个程序，输入找零的钱，输出找钱方案中最少张数的方案，比如123元，最少是1张100的，1张20的，3张1元的，一共5张！


* 分析
> 这样的题目运用的贪心策略是每次选择最大的钱，如果最后超过了，再选择次大的面值，然后次次大的面值，一直到最后与找的钱相等，这种情况大家再熟悉不过了，下面就直接看源代码：

* 代码
```
#include <iostream>
#include <cmath>
using namespace std;

int main(int argc, char* argv[]) {
  int MoneyClass[6] = {100,50,20,10,5,1}; //记录钱的面值
  int MoneyIndex [6] ={0};           //记录每种面值的数量
  int MoneyAll,MoneyCount = 0,count=0;

  cout<<"please enter the all money you want to exchange:"<<endl;
  cin>>MoneyAll;

  for(int i=0;i<6;) {    //只有这个循环才是主体
    if ( MoneyCount+MoneyClass[i] > MoneyAll) {
      i++;
      continue;
    }

    MoneyCount += MoneyClass[i];
    ++ MoneyIndex[i];
    ++ count;

    if (MoneyCount == MoneyAll)
      break;
  }

  for(i=0;i<6;++i) {     //控制输出的循环
    if (MoneyIndex[i] !=0 ) {
      switch(i) {
      case 0:
        cout<<"the 100 have:"<<MoneyIndex[i]<<endl;
        break;
      case 1:
        cout<<"the 50 have:"<<MoneyIndex[i]<<endl;
        break;
      case 2:
        cout<<"the 20 have:"<<MoneyIndex[i]<<endl;
        break;
      case 3:
        cout<<"the 10 have:"<<MoneyIndex[i]<<endl;
        break;
      case 4:
        cout<<"the 5 have:"<<MoneyIndex[i]<<endl;
        break;
      case 5:
        cout<<"the 1 have:"<<MoneyIndex[i]<<endl;
        break;
      }
    }
  }
  cout<<"the total money have:"<<count<<endl;
  system("pause");
  return 0;
}

```

### 多机调度问题
* 题目
> n个作业组成的作业集，可由m台相同机器加工处理。要求给出一种作业调度方案，使所给的n个作业在尽可能短的时间内由m台机器加工处理完成。作业不能拆分成更小的子作业；每个作业均可在任何一台机器上加工处理。

* 分析
> 这个问题是NP完全问题，还没有有效的解法(求最优解)，但是可以用贪心选择策略设计出较好的近似算法(求次优解)。当n<=m时，只要将作业时间区间分配给作业即可；当n>m时，首先将n个作业从大到小排序，然后依此顺序将作业分配给空闲的处理机。也就是说从剩下的作业中，选择需要处理时间最长的，然后依次选择处理时间次长的，直到所有的作业全部处理完毕，或者机器不能再处理其他作业为止。如果我们每次是将需要处理时间最短的作业分配给空闲的机器，那么可能就会出现其它所有作业都处理完了只剩所需时间最长的作业在处理的情况，这样势必效率较低。在下面的代码中没有讨论n和m的大小关系，把这两种情况合二为一了。


* 代码
```
#include<iostream>
#include<algorithm>
using namespace std;
int speed[10010];
int mintime[110];

bool cmp( const int &x,const int &y) {
    return x>y;
}

int main() {
  int n,m;
  memset(speed,0,sizeof(speed));
   memset(mintime,0,sizeof(mintime));
    cin>>n>>m;
     for(int i=0;i<n;++i) cin>>speed[i];
    sort(speed,speed+n,cmp);
    for(int i=0;i<n;++i) {
      *min_element(mintime,mintime+m)+=speed[i];
    }
    cout<<*max_element(mintime,mintime+m)<<endl;
}

```

### 小船过河问题
* 题目
> POJ1700是一道经典的贪心算法例题。题目大意是只有一艘船，能乘2人，船的运行速度为2人中较慢一人的速度，过去后还需一个人把船划回来，问把n个人运到对岸，最少需要多久。

* 分析
> 先将所有人过河所需的时间按照升序排序，我们考虑把单独过河所需要时间最多的两个旅行者送到对岸去，有两种方式：
> 1.最快的和次快的过河，然后最快的将船划回来；次慢的和最慢的过河，然后次快的将船划回来，所需时间为：t[0]+2*t[1]+t[n-1]；
> 2.最快的和最慢的过河，然后最快的将船划回来，最快的和次慢的过河，然后最快的将船划回来，所需时间为：2*t[0]+t[n-2]+t[n-1]。
> 算一下就知道，除此之外的其它情况用的时间一定更多。每次都运送耗时最长的两人而不影响其它人，问题具有贪心子结构的性质。


* 代码

```
#include<iostream>
#include<algorithm>
using namespace std;

int main() {
  int a[1000],t,n,sum;
  scanf("%d",&t);
  while(t--) {
    scanf("%d",&n);
    sum=0;
    for(int i=0;i<n;i++) scanf("%d",&a[i]);
    while(n>3) {
      sum=min(sum+a[1]+a[0]+a[n-1]+a[1],sum+a[n-1]+a[0]+a[n-2]+a[0]);
      n-=2;
    }
    if (n==3) sum+=a[0]+a[1]+a[2];
    else if (n==2) sum+=a[1];
    else sum+=a[0];
    printf("%d\n",sum);
  }
}
```

### 销售比赛
* 题目
> 假设有偶数天，要求每天必须买一件物品或者卖一件物品，只能选择一种操作并且不能不选，开始手上没有这种物品。现在给你每天的物品价格表，要求计算最大收益。

* 分析
> 首先要明白，第一天必须买，最后一天必须卖，并且最后手上没有物品。那么除了第一天和最后一天之外我们每次取两天，小的买大的卖，并且把卖的价格放进一个最小堆。如果买的价格比堆顶还大，就交换。这样我们保证了卖的价格总是大于买的价格，一定能取得最大收益。

* 代码

```
#include<queue>
#include<vector>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<iostream>
#include<algorithm>
using namespace std;
long long int price[100010],t,n,res;

int main() {
  ios::sync_with_stdio(false);
  cin>>t;
  while(t--) {
    cin>>n;
    priority_queue<long long int, vector<long long int>, greater<long long int> > q;
    res=0;
    for(int i=1;i<=n;i++) {
      cin>>price[i];
    }
    res-=price[1];
    res+=price[n];
    for(int i=2;i<=n-1;i=i+2) {
      long long int buy=min(price[i],price[i+1]);
      long long int sell=max(price[i],price[i+1]);
      if (!q.empty()) {
        if (buy>q.top()) {
          res=res-2*q.top()+buy+sell;
          q.pop();
          q.push(buy);
          q.push(sell);
        } else {
          res=res-buy+sell;
          q.push(sell);
        }
      } else {
        res=res-buy+sell;
        q.push(sell);
      }
    }
    cout<<res<<endl;
  }
}
```

### Huffman编码
### Dijkstra算法
### 最小生成树算法
