---
title: 五大经典算法-4.动态规划
---
# 动态规划
## 概念
> dynamic programming is a method for solving a complex problem by breaking it down into a collection of simpler subproblems.
> 动态规划是通过拆分问题，定义问题状态和状态之间的关系，使得问题能够以递推（或者说分治)的方式去解决。

## 思想
   动态规划算法的有效性依赖于待求解问题本身具有的两个重要性质：最优子结构性质和子问题重叠性质。

1 、最优子结构性质。如果问题的最优解所包含的子问题的解也是最优的，我们就称该问题具有最优子结构性质（即满足最优化原理)。最优子结构性质为动态规划算法解决问题提供了重要线索。

2 、子问题重叠性质。子问题重叠性质是指在用递归算法自顶向下对问题进行求解时，每次产生的子问题并不总是新问题，有些子问题会被重复计算多次。动态规划算法正是利用了这种子问题的重叠性质，对每一个子问题只计算一次，然后将其计算结果保存在一个表格中，当再次需要计算已经计算过的子问题时，只是在表格中简 单地查看一下结果，从而获得较高的解题效率。

当我们已经确定待解决的问题需要用动态规划算法求解时，通常可以按照以下步骤设计动态规划算法：

1 、分析问题的最优解，找出最优解的性质，并刻画其结构特征；

2 、递归地定义最优值；

3 、采用自底向上的方式计算问题的最优值；

4 、根据计算最优值时得到的信息，构造最优解。

1 ～ 3 步是动态规划算法解决问题的基本步骤，在只需要计算最优值的问题中，完成这三个基本步骤就可以了。如果问题需要构造最优解，还要执行第 4 步； 此时，在第 3 步通常需要记录更多的信息，以便在步骤 4 中，有足够的信息快速地构造出最优解。



## 题目
### 1. 最长公共子串(LCS)
* 题目

> 一个序列 S,如果分别是两个或多个已知序列的子序列，且是所有符合此条件序列中最长的，则 S 称为已知序列的最长公共子序列。
* 分析
> 转移方程：
>
> dp[i,j] = 0                               IF:   i=0 || j=0
>
> dp[i,j] = dp[i-1][j-1]+1                  IF:   i>0,j>0, a[i] = b[j]
>
> dp[i,j] = max(dp[i-1][j],dp[i][j-1])      IF:   i>0,j>0, a[i] != b[j]

* 代码

```
#include "stdio.h"
#define M 8
#define N 6


void printLSC(int i, int j,char *a, int status[][N]){
  if(i == 0 || j== 0)
    return;
  if(status[i][j] == 0){
    printLSC(i-1,j-1,a,status);
    printf("%c",a[i]);
  }else{
    if(status[i][j] == 1)
      printLSC(i-1,j,a,status);
    else
      printLSC(i,j-1,a,status);
  }
}
main(){
  int i,j;

  char a[] = {' ','A','B','C','B','D','A','B'};
  char b[] = {' ','B','D','C','B','A'};
  int status[M][N]; //保存状态
  int dp[M][N];

  for(i = 0; i < M; i++)
    for(j = 0; j < N; j++){
      dp[i][j] = 0;
      status[i][j] = 0;
    }

  for(i = 1; i < M; i++)
    for(j = 1; j < N; j++){
      if(a[i] == b[j]){
        dp[i][j] = dp[i-1][j-1] + 1;
        status[i][j] = 0;
      }
      else if(dp[i][j-1] >= dp[i-1][j]){
        dp[i][j] = dp[i][j-1];
        status[i][j] = 2;
      }
      else{
        dp[i][j] = dp[i-1][j];
        status[i][j] = 1;
      }


    }
  printf("最大长度：%d",dp[M-1][N-1]);
  printf("\n");
  printLSC(M-1,N-1,a,status);
  printf("\n");

}
```



### 2.最长递增子序列(LIS)
* 题目

> 给定一个序列 An = a1 ,a2 ,  ... , an,找出最长的子序列使得对所有 i < j,ai < aj 。


* 分析

> 转移方程：b[k]=max(max(b[j]|a[j]<a[k],j<k)+1,1);


* 代码

```
#include "stdio.h"

main(){
  int i,j,length,max=0;
  int a[] = {
    1,-1,2,-3,4,-5,6,-7
  };
  int *b;
  b = (int *)malloc(sizeof(a));
  length = sizeof(a)/sizeof(a[0]);

  for(i = 0; i < length; i++){
    b[i] = 1;
    for(j = 0; j < i; j++){
      if(a[i] > a[j] && b[i] <= b[j]){
        b[i] = b[j] + 1;
      }
    }
  }
  for(i = 0; i < length; i++)
    if(b[i] > max)
      max = b[i];

  printf("%d",max);
}

```

### 3.最大连续子序列之和
* 题目

> 给定K个整数的序列{ N1, N2, ..., NK }，其任意连续子序列可表示为{ Ni, Ni+1, ..., Nj }，其中 1 <= i <= j <= K。最大连续子序列是所有连续子序中元素和最大的一个， 例如给定序列{ -2, 11, -4, 13, -5, -2 }，其最大连续子序列为{ 11, -4, 13 }，最大和为20。

* 分析

> 状态转移方程： sum[i]=max(sum[i-1]+a[i],a[i])

* 题目

```
#include "stdio.h"

main(){
  int i,sum = 0, max = 0;
  int data[] = {
    1,-2,3,-1,7
  };
  for(i = 0; i < sizeof(data)/sizeof(data[0]); i++){
    sum += data[i];
    if(sum > max)
      max = sum;
    if(sum < 0)
      sum = 0;
  }
  printf("%d",max);
}

```


### 4.01背包问题
* 题目

> 有N件物品和一个容量为V的背包。第i件物品的费用是c[i]，价值是w[i]。求解将哪些物品装入背包可使价值总和最大。

* 分析

> 转移方程：dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]] + value[i]


* 代码

```
#include "stdio.h"
#define max(a,b) ((a)>(b)?(a):(b))

main(){

  int v = 10 ;
  int n = 5 ;

   int value[] = {0, 8 , 10 , 4 , 5 , 5};
  int weight[] = {0, 6 , 4 , 2 , 4 , 3};
  int i,j;
  int dp[n+1][v+1];
  for(i = 0; i < n+1; i++)
    for(j = 0; j < v+1; j++)
      dp[i][j] = 0;


  for(i = 1; i <= n; i++){
    for(j = 1; j <= v; j++){
      if(j >= weight[i])
        dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight[i]] + value[i]);
      else
        dp[i][j] = dp[i-1][j];
    }
  }

  printf("%d",dp[n][v]);
}
```


### 5.青蛙跳台阶问题
* 题目

> 一只青蛙可以一次跳一级台阶，也可以一次跳两级台阶，如果青蛙要跳上n级台阶，共有多少钟跳法？



* 分析

> 当青蛙即将跳上n级台阶时，共有两种可能性，一种是从n-1级台阶跳一步到n级，另外一种是从n-2级台阶跳两步到n级，所以求到n级台阶的所有可能性f(n)就转变为了求到n-2级台阶的所有可能性f(n-2)和到n-1级台阶的所有可能性f(n-1)之和，以此类推至最后f(2)=f(0)+f(1)=1+1。递推公式就是f(n) = f(n - 1) + f(n - 2)


* 代码

```
public class Fibonacci {
  public int fibonacci(int n) {
    int[] dp = { 1, 1, 0 };
    if (n < 2) {
      return 1;
    }
    for (int i = 2; i <= n; i++) {
      //递推公式f(n) = f(n - 1) + f(n -2)
      dp[2] = dp[0] + dp[1];
      dp[0] = dp[1];
      dp[1] = dp[2];
    }
    return dp[2];
  }

  public static void main(String[] args) {
    Fibonacci fb = new Fibonacci();
    for (int i = 0; i < 10; i++) {
      System.out.print(fb.fibonacci(i));
      System.out.print(" ");
    }

  }
}
```

* 相关题目 -- 青蛙变态跳台阶问题

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法？



### 6.收集苹果
* 题目

> 平面上有N*M个格子，每个格子中放着一定数量的苹果。你从左上角的格子开始，每一步只能向下走或是向右走，每次走到一个格子上就把格子里的苹果收集起来，这样下去，你最多能收集到多少个苹果。
> 输入：
> 第一行输入行数和列数
> 然后逐行输入每个格子的中的苹果的数量
> 输出：
> 最多能收到的苹果的个数。


* 分析

> 这是一个典型的二维数组DP问题
> 基本状态：
> 当你到达第x行第y列的格子的时候，收集到的苹果的数量dp[x][y]。
> 转移方程：
> 由于你只能向右走或者向下走，所以当你到达第x行第y列的格子的时候，你可能是从第x-1行第y列或者第x行第y-1列到达该格子的，而我们最后只要收集苹果最多的那一种方案。
> 所以：
> dp[x][y] = max( if(x>0) dp[x-1][y] , if(y>0) dp[x][y-1])


* 题目

```
#include<iostream>
#include<string.h>
using namespace std;
int a[100][100];
int dp[100][100];
int m,n;

void dp_fun(int x,int y)
{
  dp[x][y] = a[x][y];
  int max = 0;
  if(x > 0 && max < dp[x-1][y])
  {
    max = dp[x-1][y];
  }
  if(y > 0 && max < dp[x][y-1])
  {
    max = dp[x][y-1];
  }
  dp[x][y] += max;
  if(x<m-1)
  {
    dp_fun(x+1,y);
  }
  if(y<n-1)
  {
    dp_fun(x,y+1);
  }
  return;
}

int main()
{
  memset(dp,0,sizeof(dp));
  cin>>m>>n;
  for(int i=0;i<m;i++)
  {
    for(int j=0;j<n;j++)
    {
      cin>>a[i][j];
    }
  }
  dp_fun(0,0);
  for(int i=0;i<m;i++)
  {
    for(int j=0;j<n;j++)
    {
      cout<<dp[i][j]<<"\t";
    }
    cout<<endl;
  }
  return 0;
}

```


### 7.数塔取数问题
* [题目](http://www.cnblogs.com/DiaoCow/archive/2010/04/18/1714859.html)

> 一个高度为N的由正整数组成的三角形，从上走到下，求经过的数字和的最大值。
> 每次只能走到下一层相邻的数上，例如从第3层的6向下走，只能走到第4层的2或9上。
> 5
> 8 4
> 3 6 9
> 7 2 9 5
> 例子中的最优方案是：5 + 8 + 6 + 9 = 28。

* 分析
> 站在位置9，我们可以选择沿12方向移动，也可以选择沿着15方向移动，现在我们假设“已经求的”沿12方向的最大值x和沿15方向的最大值y，那么站在9的最大值必然是：Max(x,y) + 9。
> 因此不难得出，对于任意节点i,其状态转移方程为：m[i] = Max(a[i的左孩子] , a[i的右孩子]) + a[i]
> 首先什么是“数塔类型”？从某一点转向另一点或者说是从某一状态转向另一状态，有多种选择方式（比如这里的9->12 , 9->15)，从中选取一条能产生最优值的路径。
> 这类问题的思考方法：假设后续步骤的结果已知，比如这里假设已经知道沿12方向的最大值x和沿15方向的最大值y。


* 代码

```
#include    <stdio.h>

#define        N    10000
#define        Max(a,b)    ((a) > (b) ? (a) : (b))

int     a[N];

int main(void)
{
    int        n , m , i , k , j;

    scanf("%d",&m);
    while(m-- > 0)
    {
        scanf("%d",&n);
        k = (1 + n) * n / 2;
        for(i = 1 ; i <= k; i++)
        {
            scanf("%d",a+i);
        }

        k = k - n;
        for(i = k , j = 0 ; i >= 1 ; i--)
        {
            a[i] = a[i] + Max(a[i+n],a[i+n-1]);
            if(++j == n -1)
            {
                n--;
                j = 0;
            }
        }
        printf("%d\n",a[1]);

    }

    return    0;
}
```

### 8.免费馅饼问题
* 题目


> 都说天上不会掉馅饼，但有一天gameboy正走在回家的小径上，忽然天上掉下大把大把的馅饼。说来gameboy的人品实在是太好了，这馅饼别处都不掉，就掉落在他身旁的10米范围内。馅饼如果掉在了地上当然就不能吃了，所以gameboy马上卸下身上的背包去接。但由于小径两侧都不能站人，所以他只能在小径上接。由于gameboy平时老呆在房间里玩游戏，虽然在游戏中是个身手敏捷的高手，但在现实中运动神经特别迟钝，每秒种只有在移动不超过一米的范围内接住坠落的馅饼。现在给这条小径如图标上坐标：
![](http://p15w49jjb.bkt.clouddn.com/x7crw.gif)

> 为了使问题简化，假设在接下来的一段时间里，馅饼都掉落在0-10这11个位置。开始时gameboy站在5这个位置，因此在第一秒，他只能接到4,5,6这三个位置中期中一个位置上的馅饼。问gameboy最多可能接到多少个馅饼？（假设他的背包可以容纳无穷多个馅饼)
>
> Input
> 输入数据有多组。每组数据的第一行为以正整数n(0<n<100000)，表示有n个馅饼掉在这条小径上。在结下来的n行中，每行有两个整数x,T(0<T<100000),表示在第T秒有一个馅饼掉在x点上。同一秒钟在同一点上可能掉下多个馅饼。n=0时输入结束。
>
> Output
> 每一组输入数据对应一行输出。输出一个整数m，表示gameboy最多可能接到m个馅饼。
> 提示：本题的输入数据量比较大，建议用scanf读入，用cin可能会超时。
>
> Sample Input
> 6 5 1 4 1 6 1 7 2 7 2 8 3 0
>
> Sample Output
> 4

* 分析

> 类似于DP中的数塔，不过要倒过来算，从下往上算，最后输出初始位置的数即可， 为了便于判断边界，可以将数组宽度开大一些，让它从1~11计数，这样就不用单独计算边界了， 如果数塔不懂，可以看我之前发的经典数塔题。

* 代码

```
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ARRAY_SIZE 100000
int data[MAX_ARRAY_SIZE][11];//存放最初的数据
int cost[MAX_ARRAY_SIZE][11];//存放各个子问题的最优解
int mark[MAX_ARRAY_SIZE][11];//存放输出最优解方案标志
int main(int argc,char *argv[])
{
  int n;
  while(scanf("%d",&n),n!=0){
    memset(data,0,sizeof(data));
    int i,x,T,max_T=0;
    //初始化data
    for(i=1;i<=n;i++){
      scanf("%d%d",&x,&T);
      if(T>max_T)
        max_T=T;
      data[T][x]++;
    }
    //dp初始化
    for(i=0;i<11;i++){
      cost[max_T][i]=data[max_T][i];
    }
    //dp过程
    for(i=max_T-1;i>=0;i--){
      int j;
      for(j=0;j<11;j++){
        int lvalue,mvalue,rvalue,maxvalue;
        if(j==0){
          lvalue=-1;
        }else{
          lvalue=cost[i+1][j-1];
        }
        mvalue=data[i+1][j];
        if(j==10){
          rvalue=-1;
        }else{
          rvalue=cost[i+1][j+1];
        }
        if(lvalue>mvalue){
          maxvalue=lvalue;
          mark[i][j]=-1;
        }else{
          if(mvalue>rvalue){
            maxvalue=mvalue;
            mark[i][j]=0;
          }else{
            maxvalue=rvalue;
            mark[i][j]=1;
          }
        }
        cost[i][j]=data[i][j]+maxvalue;
      }
    }
    printf("%d\n",cost[0][5]);
  }
  return 0;
}
```
