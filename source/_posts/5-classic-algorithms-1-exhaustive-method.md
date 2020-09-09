---
title: 五大经典算法|1.穷举法
date: 2020-04-10
categories:
- 五大经典算法
tags:
- 穷举法
---
穷举法，又称枚举法，或称为暴力破解法.

经典例题：素数判断,鸡兔同笼,最大连续子序列,火柴棍等式,计算几何距离

<!-- more -->

## 基本概念
穷举法的基本思想是根据题目的部分条件确定答案的大致范围，并在此范围内对所有可能的情况逐一验证，直到全部情况验证完毕。若某个情况验证符合题目的全部条件，则为本问题的一个解；若全部情况验证后都不符合题目的全部条件，则本题无解。

从计算机的有限集合中，对每一个可能解进行判断，直到找到正确的答案。使用穷举法主要是要获取有限集合，然后一一枚举获取需要的答案。


## 算法思想
穷举法的基本思想就是从所有可能的情况中搜索正确的答案，其执行步骤如下：

1.对于一种可能的情况，计算其结果。

2.判断结果是否符合要求，如果不满足则执行第1步来搜索下一个可能的情况；如果符合要求，则表示寻找到一个正确答案。

在使用穷举法时，需要明确问题的答案的范围，这样才可以在指定的范围内搜索答案。指定范围之后，就可以使用循环语句和条件语句逐步验证候选答案的正确性，从而得到需要的正确答案。


## 经典例题
### 素数判断
问题：找出一个区间[100,200]内部的素数.

> 如果一个常数n不能整除任意一个大于2并且小于n的整数，那么这个数就称之为素数。

我们来分析一下这个问题，第一题目已经明确给定了集合区间，第二题目已经给定判定条件，我们只需要遍历集合区间内的所有常数，执行一片判断条件，就能找出所有的素数，求出题目的答案，所以满足使用穷举算法的条件。

<details>
  <summary>穷举法求解素数判断问题C语言实现代码</summary>

```C
#include "stdio.h"

bool checkPrime(int number){
  for (int i=2; i<number; i++){
    if (number%i == 0){
      return false;
    }
  }
  return true;
}

int main() {
  int min, max, t;
  printf("输入最小数:\n");
  scanf("%d", &min);
  printf("输入最大数:\n");
  scanf("%d", &max);
  if (min>max) {
    printf("输入数据有误!\n");
    return 1;
  }

  printf("区间范围%d~%d的素数为:\n", min, max);
  for (int i=min; i<=max; i++) {
    if (checkPrime(i)) {
      printf("%d\n", i);
    }
  }
  return 0;
}
```

</details>


### 鸡兔同笼
鸡兔同笼问题最早记载中1500年前的《孙子算经》，这是我国古代一个非常有名的问题。鸡兔同笼问题的原文如下：今天鸡兔同笼，上有三十五个头，下有九十四足，问鸡兔各几何？这个问题的大致意思是：在一个笼子里关着若干只鸡和若干只兔，从上面看共有35个头，从下面数共有94只脚。问笼中鸡和兔的数量各是多少？

这个问题需要计算鸡的数量和兔的数量，我们通过分析可以知道鸡的数量应该在1~35之间。这样我们可以使用穷举法来逐个判断是否符合，从而搜索答案。


<details>
  <summary>穷举法求解鸡兔同笼问题C语言实现代码</summary>

```C
#include<iostream>
using namespace std;
/*
输入参数head是笼中头的总数，foot是笼中脚的总数，chicken是鸡的总数,rabbit是兔的总数
返回结果为0，表示没有搜索到符合条件的结果；
返回结果为1，表示搜索到了符合条件的结果
*/
int qiongju(int head, int foot, int *chicken, int *rabbit) {
  int re,i,j;
  re=0;
  for (i=0;i<=head;i++) {   //进行循环
    j=head-i;
    if (i*2+j*4==foot) {    //进行判断
      re=1;        //找到答案
      *chicken=i;
      *rabbit=j;
    }
  }
  return re;
}

int main() {
  int chicken,rabbit,head,foot;
  cout<<"穷举法求解鸡兔同笼问题："<<endl;
  cout<<"请输入头数：";
  cin>>head;
  cout<<"请输入脚数：";
  cin>>foot;
  int res=qiongju(head,foot,&chicken,&rabbit);
  if (res==1) {
    cout<<"鸡有"<<chicken<<"只，兔有"<<rabbit<<"只。"<<endl;
  } else {
    cout<<"无法求解！"<<endl;
  }
  return 0;
}
```

</details>

### 最大连续子序列
给定K个整数的序列{ N_1, N_2, ..., N_K}，其任意连续子序列可表示为{ Ni, Ni+1, ..., Nj}，其中 1<=i<=j<=K。最大连续子序列是所有连续子序列中元素和最大的一个，例如给定序列{-2, 11, -4, 13, -5, -2 }，其最大连续子序列为{11, -4, 13}，最大和为20。

> Input: 测试输入包含若干测试用例，每个测试用例占2行，第1行给出正整数K(<10000)，第2行给出K个整数，中间用空格分隔。当K为0时，输入结束，该用例不被处理。

> Output: 对每个测试用例，在1行里输出最大和、最大连续子序列的第一个和最后一个元素，中间用空格分隔。如果最大连续子序列不唯一，则输出序号i和j最小的那个(如输入样例的第2、3组)。若所有K个元素都是负数，则定义其最大和为0，输出整个序列的首尾元素。

<details>
  <summary>穷举法求解最大连续子序列C语言实现代码</summary>

```C
#include <iostream>
using namespace std;

int main() {
  int a[200] = {0};   // 数组a记录整数序列
  // count记录负数个数
  // max 最大和  max_f 最大和最前端 max_l最大和最后端
  int n, i, j, s, count, max, max_f, max_l;

  // 以输入作为循环条件实现多组数据的输入
  while(cin>>n) {
    if ( n == 0 ) return 0;

    // 特殊情况的判断
    count = 0;
    for ( i = 0; i < n; ++i ) {
      cin>>a[i];
      if ( a[i] < 0 ) count++; //记录负数个数
    }

    if (count == n) {
      max = 0;
      max_f = a[0];
      max_l = a[n-1];
    } else { // 大多数情况的操作
      max = a[0];max_f = a[0];max_l = a[0];
      for ( i = 0; i < n; ++i ) { // 从a[0]开始计算各情况
        s = a[i];
        for ( j = i+1; j < n; ++j ) { // 算法 可草稿推演
          s += a[j];
          if (s>max) { //寻找最大
            max = s;
            max_f = a[i];
            max_l = a[j];
          }
        }
      }
    }
    // 输出结果
    cout<<max<<" "<<max_f<<" "<<max_l<<endl;
  }
  return 0;
}
```

</details>

### 火柴棍等式
给你n根火柴棍，你可以拼出多少个形如“A+B=C”的等式？等式中的A、B、C是用火柴棍拼出的整数(若该数非零，则最高位不能是0)。

用火柴棍拼数字0-9的拼法如图所示：

![火柴棍等式](https://imzhanghao.oss-cn-qingdao.aliyuncs.com/img/火柴棍等式.png)

**注意**

1.加号与等号各自需要两根火柴棍

2.如果A≠B，则A+B=C与B+A=C视为不同的等式(A、B、C>=0)

3.n根火柴棍必须全部用上

**输入格式**： 输入共一行，又一个整数n(n<=24)

**输出格式**： 输出共一行，表示能拼成的不同等式的数目

**样例1**

输入： 14， 输出： 2 
> 2个等式为：0+1=1和1+0=1。

**样例2**

输入： 18， 输出： 9
> 9个等式为：0+4=4， 0+11=11， 1+10=11， 2+2=4， 2+7=9， 4+0=4， 7+2=9， 10+1=11， 11+0=11

其实是一道很简单的枚举题。

首先看范围，火柴棒的个数小于等于24，减去加号、等号后只有二十根。

再看每位数字需要的火柴棒数目，发现1最少，只要两根。

那么尽量多添1使得组成的数尽可能大，发现当填到1111时，火柴棒组成基本超过24，故我们大致找到一个范围小于等于1111。

最后只需在0~1111内枚举两个数字，使得它们和它们的和组成的火柴棒个数为n。

<details>
  <summary>穷举法求解火柴棍等式问题C语言实现代码</summary>

```C
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
int num[100],ans1,i,j,n,l;
int main() {
	scanf("%d",&n);
	num[0]=6; num[1]=2; num[2]=5; num[3]=5; num[4]=4;
	num[5]=5; num[6]=6; num[7]=3; num[8]=7; num[9]=6;
	n-=4;
	
  if (n<9)  { cout<<"0"<<endl; return 0; }

	for (i=0;i<=1001;i++) {
	  for (j=0;j<=i;j++) {
	  	int x=i-j;
	  	char s1[10],s2[10],s3[10];
	  	sprintf(s1,"%d",i);
	  	sprintf(s2,"%d",j);
	  	sprintf(s3,"%d",x);
	  	int ans=0;
	  	for (l=1;l<=strlen(s1);l++) 
	  	  ans+=num[s1[l-1]-48];
	  	for (l=1;l<=strlen(s2);l++) 
	  	  ans+=num[s2[l-1]-48];
		  for (l=1;l<=strlen(s3);l++)
	  	  ans+=num[s3[l-1]-48]; 
	  	if (ans==n)
	  	  ans1++;
	  }
  }
	printf("%d",ans1);
	return 0;
}

```

</details>

### 计算几何距离
今天小明考完了期末考试,他在教学楼里闲逛,他看着教学楼里一间间的教室,于是开始思考:如果从一个坐标为 (x1,y1,z1)的教室走到(x2,y2,z2)的距离为 |x1−x2|+|y1−y2|+|z1−z2|，那么有多少对教室之间的距离是不超过R的呢?

**输入**
第一行是一个整数T(1≤T≤10), 表示有T组数据，接下来是T组数据,对于每组数据: 第一行是两个整数  n,q(1≤n≤5×104,1≤q≤103), 表示有n间教室, q次询问. 接下来是n行, 每行3个整数  xi,yi,zi(0≤xi,yi,zi≤10),表示这间教室的坐标. 最后是q行,每行一个整数R(0≤R≤109).

**输出**
对于每个询问RR输出一行一个整数,表示有多少对教室满足题目所述的距离关系.

**样例输入** 
```
1 
3 3 
0 0 0 
1 1 1 
1 1 1 
1 2 3 
```
**样例输出** 
```
1 
1 
3
```
> 对于样例,1号教室和2号教室之间的距离为3, 1号和3号之间的距离为3, 2号和3号之间的距离为0

题意：在一个三维空间中有N个点，q次查询，每次查询给一距离r，求出三维空间中有多少对点之间的哈密顿距离小于r。

思路：一开始的时候如果按照朴素的想法，先离线处理，两两配对求出每两个点之间的距离，之后输出，但是本题中点的数目n的数据较大，如果要全部处理的话需要109左右的操作数，肯定会超时。那么这个时候我们仔细观察后发现，每一个点的范围很小，0<=x,y,z<=10，如果我们通过坐标来遍历每一个点，那么就只需要10^3的复杂度，显然更合适。所以本题也是如此，通过以坐标为单位的枚举，就可以得到最后的结果.

<details>
  <summary>穷举法求解几何距离问题C语言实现代码</summary>

```C
#include <bits/stdc++.h>
using namespace std;
typedef long long LL;
const int MAX = 10005;
const int MOD = 1e9+7;
const int INF = 0x3f3f3f3f;

int n, q, t, tem;
int a, b, c, x, y, z;
LL aa[35];
LL dex[15][15][15];

int dis(int aa, int bb, int cc, int xx, int yy, int zz) {
  return abs(aa-xx)+abs(bb-yy)+abs(cc-zz);
}

int main() {
  scanf("%d",&t);
  while(t--) {
    memset(aa, 0, sizeof(aa));
    memset(dex, 0, sizeof(dex));
    scanf("%d%d",&n,&q);
    while(n--) {
      scanf("%d%d%d",&x,&y,&z);
      ++dex[x][y][z];
    }
    for (a = 0; a <= 10; ++a)
      for (b = 0; b <= 10; ++b)
        for (c = 0; c <= 10; ++c)
          if (dex[a][b][c])
            for (x = 0; x <= 10; ++x)
              for (y = 0; y <= 10; ++y)
                for (z = 0; z <= 10; ++z)
                  if (dex[x][y][z]) {
                    tem = dis(a, b, c, x, y, z);
                    if (tem == 0)
                      aa[tem] += (dex[x][y][z])*(dex[x][y][z]-1)/2;
                    else
                      aa[tem] += dex[x][y][z]*dex[a][b][c];
                  }
    for (int i = 1; i <= 30; ++i)
      aa[i] /= 2;
    for (int i = 1; i <= 30; ++i)
      aa[i] += aa[i-1];
    while(q--) {
      scanf("%d",&tem);
      if (tem > 30)
        tem = 30;
      printf("%lld\n",aa[tem]);
    }
  }
  return 0;
}
```

</details>


### 备选题目
* 换分币：用一元人民币兑换成1分、2分和5分硬币，有多少种不同的兑换方法？请输出所有可能的方案。
* 年龄几何：张三、李四、王五、刘六的年龄成一等差数列，他们四人的年龄相加是26，相乘是880，求以他们的年龄为前4项的等差数列的前20项
* 三色球问题：若一个口袋中放有12个球，其中有3个红的。3个白的和6个黒的，问从中任取8个共有多少种不同的颜色搭配？
