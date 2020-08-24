---
title: 五大经典算法|1.穷举法
date: 2020-04-10
categories:
- 五大经典算法
tags:
- 穷举法
---
> 穷举法，又称枚举法，或称为暴力破解法.
> 经典例题：1)素数判断,2)鸡兔同笼,3)最大连续子序列,4)火柴棍等式,5)计算几何距离,6)计算几何,7)备选题目
<!-- more -->

## 概念
穷举法的基本思想是根据题目的部分条件确定答案的大致范围，并在此范围内对所有可能的情况逐一验证，直到全部情况验证完毕。若某个情况验证符合题目的全部条件，则为本问题的一个解；若全部情况验证后都不符合题目的全部条件，则本题无解。

从计算机的有限集合中，对每一个可能解进行判断，直到找到正确的答案。使用穷举法主要是要获取有限集合，然后一一枚举获取需要的答案。

## 思想
* 穷举法的基本思想就是从所有可能的情况中搜索正确的答案，其执行步骤如下：
  (1)对于一种可能的情况，计算其结果。
  (2)判断结果是否符合要求，如果不满足则执行第(1)步来搜索下一个可能的情况；如果符合要求，则表示寻找到一个正确答案。

在使用穷举法时，需要明确问题的答案的范围，这样才可以在指定的范围内搜索答案。指定范围之后，就可以使用循环语句和条件语句逐步验证候选答案的正确性，从而得到需要的正确答案。


## 题目
### 素数判断
* 题目
> 判断一个区间[100,200]内部的素数.
>   - 1 给定集合区间.
>   - 2 给定判定条件.
> 所以满足使用穷举算法的条件

* 分析
> 例 常数n
> 如果n=1 或 n=2
> 或 n “不能整除任意 一个大于2并且小于n的整数”，那么这个数就称之为素数。


* 代码

```
#include "stdio.h"

/*素数验证*/
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

### 鸡兔同笼
* 题目
> 鸡兔同笼问题最早记载中1500年前的《孙子算经》，这是我国古代一个非要有名的问题。
> 鸡兔同笼问题的原文如下：今天鸡兔同笼，上有三十五个头，下有九十四足，问鸡兔各几何？
> 这个问题的大致意思是：在一个笼子里关着若干只鸡和若干只兔，从上面看共有35个头，从下面数共有94只脚。
> 问笼中鸡和兔的数量各是多少？


* 分析
> 这个问题需要计算鸡的数量和兔的数量，我们通过分析可以知道鸡的数量应该在1~35之间。
> 这样我们可以使用穷举法来逐个判断是否符合，从而搜索答案。


* 代码

```
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

### 最大连续子序列
* [题目](https://blog.csdn.net/MadBam_boo/article/details/50867986)


> 给定K个整数的序列{ N1, N2, ..., NK }，其任意连续子序列可表示为{ Ni, Ni+1, ..., Nj }，其中 1 <= i <= j <= K。最大连续子序列是所有连续子序列中元素和最大的一个，例如给定序列{ -2, 11, -4, 13, -5, -2 }，其最大连续子序列为{ 11, -4, 13 }，最大和为20。
>
> Input
> 测试输入包含若干测试用例，每个测试用例占2行，第1行给出正整数K(<10000)，第2行给出K个整数，中间用空格分隔。
> 当K为0时，输入结束，该用例不被处理。
>
> Output
> 对每个测试用例，在1行里输出最大和、最大连续子序列的第一个和最后一个元素，中间用空格分隔。
> 如果最大连续子序列不唯一，则输出序号i和j最小的那个(如输入样例的第2、3组)。
> 若所有K个元素都是负数，则定义其最大和为0，输出整个序列的首尾元素。
>
> Sample Input
> 6
> -2 11 -4 13 -5 -2
> 10
> -10 1 2 3 4 -5 -23 3 7 -21
> 6
> 5 -8 3 2 5 0
> 1
> 10
> 3
> -1 -5 -2
> 3
> -1 0 -2
> 0
>
> Sample Output
> 20 11 13
> 10 1 4
> 10 3 5
> 10 10 10
> 0 -1 -2
> 0 0 0 (最后一组数据有误)

* 分析


* 代码

```
#include <iostream>
using namespace std;

int main()
{
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
            if ( a[i] < 0 ) count++;     //记录负数个数
        }

        if (count == n) {
            max = 0;
            max_f = a[0];
            max_l = a[n-1];
        }
        // 大多数情况的操作
        else {
            max = a[0];max_f = a[0];max_l = a[0];
            for ( i = 0; i < n; ++i ) {
                //从a[0]开始计算各情况
                s = a[i];
                // 算法 可草稿推演
                for ( j = i+1; j < n; ++j ) {
                    s += a[j];
                    //寻找最大
                    if (s>max) {
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

### 火柴棍等式
* [题目](http://www.tsinsen.com/A1167)


> 给你n根火柴棍，你可以拼出多少个形如“A+B=C”的等式？等式中的A、B、C是用火柴棍拼出的整数(若该数非零，则最高位不能是0)。用火柴棍拼数字0-9的拼法如图所示：
![](http://p15w49jjb.bkt.clouddn.com/odgch.GIF)

> 注意：
> (1)加号与等号各自需要两根火柴棍
> (2)如果A≠B，则A+B=C与B+A=C视为不同的等式(A、B、C>=0)
> (3)n根火柴棍必须全部用上
>
> 输入格式：输入共一行，又一个整数n(n<=24)。
> 输出格式：输出共一行，表示能拼成的不同等式的数目。
> 样例输入1:14
> 样例输出1:2
> 样例输入2:18
> 样例输出2:9




* 分析

> 【输入输出样例1解释】
> 2个等式为0+1=1和1+0=1。
> 【输入输出样例2解释】
> 9个等式为：
> 0+4=4
> 0+11=11
> 1+10=11
> 2+2=4
> 2+7=9
> 4+0=4
> 7+2=9
> 10+1=11
> 11+0=11
>
> 其实是一道很简单的枚举题。
> 首先看范围，火柴棒的个数小于等于24，减去加号、等号后只有二十根。
> 再看每位数字需要的火柴棒数目，发现1最少，只要两根。
> 那么尽量多添1使得组成的数尽可能大，发现当填到1111时，火柴棒组成基本超过24，故我们大致找到一个范围小于等于1111。
> 最后只需在0~1111内枚举两个数字，使得它们和它们的和组成的火柴棒个数为n。

* 代码

```
#include<cstdio>
#define maxn 1000
using namespace std;

int a[10]={6,2,5,5,4,5,6,3,7,6};

int get(int x)
{
  int sum=0;
  if (x==0)return a[0];
  while(x>0)sum+=a[x%10],x/=10;
  return sum;
}

int main()
{
  int i,j,k,n,ans=0;
  scanf("%d",&n),n-=4;
  for (i=0;i<=maxn;i++)
    if (get(i)+get(i)+get(i+i)==n)ans++;

  for (i=0;i<=maxn;i++)
    for (j=0;j<=maxn;j++)if (i!=j)
      if (get(i)+get(j)+get(i+j)==n)ans++;
  printf("%d\n",ans);
  return 0;
}

```

### 计算几何距离
* [题目](https://blog.csdn.net/xinxiaxindong/article/details/75286893)

> 今天HHHH考完了期末考试,他在教学楼里闲逛,他看着教学楼里一间间的教室,于是开始思考:
> 如果从一个坐标为 (x1,y1,z1)(x1,y1,z1)的教室走到(x2,y2,z2)(x2,y2,z2)的距离为 |x1−x2|+|y1−y2|+|z1−z2||x1−x2|+|y1−y2|+|z1−z2|
> 那么有多少对教室之间的距离是不超过RR的呢?
>
> INPUT
> 第一行是一个整数T(1≤T≤10)T(1≤T≤10), 表示有TT组数据 接下来是TT组数据,对于每组数据: 第一行是两个整数  n,q(1≤n≤5×104,1≤q≤103)n,q(1≤n≤5×104,1≤q≤103), 表示有nn间教室, qq次询问. 接下来是nn行, 每行3个整数  xi,yi,zi(0≤xi,yi,zi≤10)xi,yi,zi(0≤xi,yi,zi≤10),表示这间教室的坐标. 最后是qq行,每行一个整数R(0≤R≤109)R(0≤R≤109),意思见描述.
>
> OUTPUT
> 对于每个询问RR输出一行一个整数,表示有多少对教室满足题目所述的距离关系.
> SAMPLE INPUT
> 1 3 3 0 0 0 1 1 1 1 1 1 1 2 3
> SAMPLE OUTPUT
> 1 1 3
> HINT
> 对于样例,1号教室和2号教室之间的距离为3, 1号和3号之间的距离为3, 2号和3号之间的距离为0

* 分析

> 题意：在一个三维空间中有N个点，q次查询，每次查询给一距离r，求出三维空间中有多少对点之间的哈密顿距离小于r。
>
> 思路：一开始的时候如果按照朴素的想法，先离线处理，两两配对求出每两个点之间的距离，之后输出，但是本题中点的数目n的数据较大，如果要全部处理的话需要109左右的操作数，肯定会超时。那么这个时候我们仔细观察后发现，每一个点的范围很小，0<=x,y,z<=10，如果我们通过坐标来遍历每一个点，那么就只需要10^3的复杂度，显然更合适。所以本题也是如此，通过以坐标为单位的枚举，就可以得到最后的结果.



* 代码

```
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


### 计算几何
* [题目](https://blog.csdn.net/u012596172/article/details/42553925)

> Problem Description
> E-pang Palace was built in Qin dynasty by Emperor Qin Shihuang in Xianyang, Shanxi Province. It was the largest palace ever built by human. It was so large and so magnificent that after many years of construction, it still was not completed. Building the great wall, E-pang Palace and Qin Shihuang's tomb cost so much labor and human lives that people rose to fight against Qin Shihuang's regime.
>
> Xiang Yu and Liu Bang were two rebel leaders at that time. Liu Bang captured Xianyang -- the capital of Qin. Xiang Yu was very angry about this, and he commanded his army to march to Xianyang. Xiang Yu was the bravest and the strongest warrior at that time, and his army was much more than Liu Bang's. So Liu Bang was frighten and retreated from Xianyang, leaving all treasures in the grand E-pang Palace untouched. When Xiang Yu took Xianyang, he burned E-pang Palce. The fire lasted for more than three months, renouncing the end of Qin dynasty.
>
> Several years later, Liu Bang defeated Xiangyu and became the first emperor of Han dynasty. He went back to E-pang Palace but saw only some pillars left. Zhang Liang and Xiao He were Liu Bang's two most important ministers, so Liu Bang wanted to give them some awards. Liu Bang told them: "You guys can make two rectangular fences in E-pang Palace, then the land inside the fences will belongs to you. But the corners of the rectangles must be the pillars left on the ground, and two fences can't cross or touch each other."
>
> To simplify the problem, E-pang Palace can be consider as a plane, and pillars can be considered as points on the plane. The fences you make are rectangles, and you MUST make two rectangles. Please note that the rectangles you make must be parallel to the coordinate axes.
>
> The figures below shows 3 situations which are not qualified(Thick dots stands for pillars):
>
>
>
> Zhang Liang and Xiao He wanted the total area of their land in E-pang Palace to be maximum. Please bring your computer and go back to Han dynasty to help them so that you may change the history.
>
>
> Input
> There are no more than 15 test case.
>
> For each test case:
>
> The first line is an integer N, meaning that there are N pillars left in E-pang Palace(4 <=N <= 30).
>
> Then N lines follow. Each line contains two integers x and y (0 <= x,y <= 200), indicating a pillar's coordinate. No two pillars has the same coordinate.
>
> The input ends by N = 0.
>
> Output
> For each test case, print the maximum total area of land Zhang Liang and Xiao He could get. If it was impossible for them to build two qualified fences, print "imp".
>
> Sample Input
> 8 0 0 1 0 0 1 1 1 0 2 1 2 0 3 1 3 8 0 0 2 0 0 2 2 2 1 2 3 2 1 3 3 3 0
>
>
> Sample Output
> 2 imp

* 分析
> 题意 ：告诉你ｎ个点的坐标，用其中的八个点作为顶点组成两个矩形(俩矩阵不能相交)。输出俩矩阵覆盖的面积的最大值。　
> 思路 ：因为ｎ很小，可以暴力枚举，将能够组成的矩形储存起来，然后再枚举矩形从而求出最大值(注意内含的情况)。

* 代码

```
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;
const int maxn=35;

struct point {
  int x,y;
  point() {}
  point(int xx,int yy):x(xx),y(yy) {}
} tp[maxn];

struct rectangular {
  point a,b,c,d;
  int S;
  rectangular() {}
  rectangular(point aa,point bb,point cc,point dd,int ss):a(aa),b(bb),c(cc),d(dd),S(ss) {}
};

vector <rectangular> v;
int n;

bool cmp(point p,point q) {
  if (p.x==q.x)   return p.y<q.y;
  return p.x<q.x;
}

void initial() {
  v.clear();
}

void input() {
  for (int i=0; i<n; i++)  scanf("%d %d",&tp[i].x,&tp[i].y);
  sort(tp,tp+n,cmp);
}

bool judge(point aa,point bb,point cc,point dd) {
  if (aa.x==bb.x && aa.y==cc.y && bb.y==dd.y && cc.x==dd.x)  return true;
  return false;
}

int Area(point aa,point bb,point cc,point dd) {
  int tx=bb.y-aa.y,ty=cc.x-aa.x;
  return tx*ty;
}

void get_rectangular() {
  for (int i=0; i<n; i++)
    for (int j=i+1; j<n; j++)
      for (int k=j+1; k<n; k++)
        for (int t=k+1; t<n; t++)
          if (judge(tp[i],tp[j],tp[k],tp[t]))
            v.push_back(rectangular(tp[i],tp[j],tp[k],tp[t],Area(tp[i],tp[j],tp[k],tp[t])));
}

int In(point aa,rectangular bb) {
  if (aa.x>bb.a.x && aa.x<bb.d.x && aa.y>bb.a.y && aa.y<bb.d.y)  return 2;
  if (aa.x>=bb.a.x && aa.x<=bb.d.x && aa.y>=bb.a.y && aa.y<=bb.d.y)  return 1;
  return 0;
}

int get_connect(int p,int q) {
   rectangular Min,Max;
   if (v[p].S<v[q].S)  Min=v[p],Max=v[q];
   else  Min=v[q],Max=v[p];
   int aa=In(Min.a,Max),bb=In(Min.b,Max),cc=In(Min.c,Max),dd=In(Min.d,Max);
   if (aa==2 && bb==2 && cc==2 && dd==2)  return 2;
   if (aa==0 && bb==0 && cc==0 && dd==0)  return 1;
   return 0;
}

void solve() {
  get_rectangular();
  int cnt=v.size(),ans=-1;
  for (int i=0;i<cnt;i++)
    for (int j=i+1;j<cnt;j++) {
      int num=get_connect(i,j);
      if (num==2)        ans=max(ans,max(v[i].S,v[j].S));
      else if (num==1)   ans=max(ans,v[i].S+v[j].S);
    }
  if (ans==-1)  printf("imp\n");
  else printf("%d\n",ans);
}

int main() {
  while(scanf("%d",&n)!=EOF) {
    if (n==0)  break;
    initial();
    input();
    solve();
  }
  return 0;
}
```

### 备选题目
* 换分币：用一元人民币兑换成1分、2分和5分硬币，有多少种不同的兑换方法？请输出所有可能的方案。
* 年龄几何：张三、李四、王五、刘六的年龄成一等差数列，他们四人的年龄相加是26，相乘是880，求以他们的年龄为前4项的等差数列的前20项
* 三色球问题：若一个口袋中放有12个球，其中有3个红的。3个白的和6个黒的，问从中任取8个共有多少种不同的颜色搭配？
