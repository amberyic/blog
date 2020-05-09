---
title: 五大经典算法-3.分治法
---
# 分治法
> 分治算法是一种将大问题划分为小问题的算法，理解和实现起来比较抽象，在我们的数据结构课程中，讲解得比较多，主要涉及部分，二分查找、快速排序，归并排序，二叉树相关等。

## 概念
在计算机科学中，分治法是一种很重要的算法。字面上的解释是“分而治之”，就是把一个复杂的问题分成两个或更多的相同或相似的子问题，再把子问题分成更小的子问题……直到最后子问题可以简单的直接求解，原问题的解即子问题的解的合并。这个技巧是很多高效算法的基础，如排序算法(快速排序，归并排序)，傅立叶变换(快速傅立叶变换)……

任何一个可以用计算机求解的问题所需的计算时间都与其规模有关。问题的规模越小，越容易直接求解，解题所需的计算时间也越少。例如，对于n个元素的排序问题，当n=1时，不需任何计算。n=2时，只要作一次比较即可排好序。n=3时只要作3次比较即可，…。而当n较大时，问题就不那么容易处理了。要想直接解决一个规模较大的问题，有时是相当困难的。

## 思想
分治法的设计思想是：将一个难以直接解决的大问题，分割成一些规模较小的相同问题，以便各个击破，分而治之。

分治策略是：对于一个规模为n的问题，若该问题可以容易地解决(比如说规模n较小)则直接解决，否则将其分解为k个规模较小的子问题，这些子问题互相独立且与原问题形式相同，递归地解这些子问题，然后将各子问题的解合并得到原问题的解。这种算法设计策略叫做分治法。

如果原问题可分割成k个子问题，1<k≤n，且这些子问题都可解并可利用这些子问题的解求出原问题的解，那么这种分治法就是可行的。由分治法产生的子问题往往是原问题的较小模式，这就为使用递归技术提供了方便。在这种情况下，反复应用分治手段，可以使子问题与原问题类型一致而其规模却不断缩小，最终使子问题缩小到很容易直接求出其解。这自然导致递归过程的产生。分治与递归像一对孪生兄弟，经常同时应用在算法设计之中，并由此产生许多高效算法。


## 题目

### 第1题
- 难度：Media
- 备注：需要数据结构哈希的基础知识，出自《leetcode》
- 题目描述
  Given a string containing only digits, restore it by returning all possible valid IP address combinations.
  For example:
  Given"25525511135",
  return["255.255.11.135", "255.255.111.35"]. (Order does not matter)
  https://www.nowcoder.com/practice/ce73540d47374dbe85b3125f57727e1e?tpId=46&tqId=29085&tPage=3&rp=3&ru=/ta/leetcode&qru=/ta/leetcode/question-ranking

### 第2题
- 难度：Media
- 备注：出自《leetcode》
- 题目描述
  Implementint sqrt(int x).
  Compute and return the square root of *x*.
  https://www.nowcoder.com/practice/09fbfb16140b40499951f55113f2166c?tpId=46&tqId=29109&tPage=4&rp=4&ru=/ta/leetcode&qru=/ta/leetcode/question-ranking

### 第3题
- 难度：Media
- 备注：需要用到STLvector的知识，出自《leetcode》
- 题目描述
  Given a collection of numbers, return all possible permutations.
  For example,
  [1,2,3]have the following permutations:
  [1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2], and[3,2,1].
  https://www.nowcoder.com/practice/4bcf3081067a4d028f95acee3ddcd2b1?tpId=46&tqId=29133&tPage=6&rp=6&ru=/ta/leetcode&qru=/ta/leetcode/question-ranking

### 第4题
- 难度：Media
- 备注：需要用到STLvector的知识，出自《leetcode》
- 题目描述
  Given a collection of numbers that might contain duplicates, return all possible unique permutations.
  For example,
  [1,1,2]have the following unique permutations:
  [1,1,2],[1,2,1], and[2,1,1].
  https://www.nowcoder.com/practice/a43a2b986ef34843ac4fdd9159b69863?tpId=46&tqId=29132&tPage=6&rp=6&ru=/ta/leetcode&qru=/ta/leetcode/question-ranking


### 第K大数
* 题目
> 在一个未排序的数组中找到第k大的元素，注意此言的第k大就是排序后的第k大的数，

* 分析
> 总是将要划界的数组段末尾的元素为划界元，将比其小的数交换至前，比其大的数交换至后，最后将划界元放在“中间位置”(左边小，右边大)。划界将数组分解成两个子数组(可能为空)。
>
> 设数组下表从low开始，至high结束。
> 1、 总是取要划界的数组末尾元素为划界元x，开始划界：
> a) 用j从low遍历到high-1(最后一个暂不处理)，i=low-1，如果nums[j]比x小就将nums[++i]与nums[j]交换位置.
> b) 遍历完后再次将nums[i+1]与nums[high]交换位置(处理最后一个元素);
> c) 返回划界元的位置i+1，下文称其为midpos.
> 这时的midpos位置的元素，此时就是整个数组中第N-midpos大的元素，我们所要做的就像二分法一样找到K=N-midpos的“中间位置”，即midpos=N-K.
> 2、 如果midpos==n-k，那么返回该值，这就是第k大的数。
> 3、 如果midpos>n-k，那么第k大的数在左半数组.
> 4、 如果midpos<n-k，那么第k大的数在右半数组.


* 代码

```
//思路首先：
//快排划界，如果划界过程中当前划界元的中间位置就是k则找到了
//time,o(n*lg(k)),space,o(1)
class Solution {
public:
    //对数组vec，low到high的元素进行划界，并获取vec[high]的“中间位置”
    int quickPartion(vector<int> &vec, int low,int high)
    {
        int x = vec[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++)
        {
            if (vec[j] <= x)//小于x的划到左边
                swap(vec,++i,j);
        }
        swap(vec,++i,high);//找到划界元的位置
        return i;//返回位置
    }
    //交换数组元素i和j的位置
    void swap(vector<int>& nums, int i, int j){
        int temp = nums[i];
        nums[i]=nums[j];
        nums[j]=temp;
    }
    int getQuickSortK(vector<int> &vec, int low,int high, int k)
    {
        if(low >= high) return vec[low];
        int  midpos = quickPartion(vec, low,high);   //对原数组vec[low]到vec[high]的元素进行划界
        if (midpos == vec.size() - k)      //如果midpos==n-k，那么返回该值，这就是第k大的数
            return vec[midpos];
        else if (midpos < vec.size() - k)  //如果midpos<n-k，那么第k大的数在右半数组
            return getQuickSortK(vec, midpos+1, high, k);
        else                               //如果midpos>n-k，那么第k大的数在左半数组
            return getQuickSortK(vec, low, midpos-1, k);
    }
    int findKthLargest(vector<int>& nums, int k) {
        return getQuickSortK(nums,0,nums.size()-1,k);
    }
};

```

### 循环赛日程表
* [题目](https://blog.csdn.net/u014755255/article/details/50570563)

>  设有n=2^k个运动员要进行网球循环赛。现要设计一个满足以下要求的比赛日程表：
>
> (1)每个选手必须与其他n-1个选手各赛一次；
> (2)每个选手一天只能参赛一次；
> (3)循环赛在n-1天内结束。
> 请按此要求将比赛日程表设计成有n行和n-1列的一个表。在表中的第i行，第j列处填入第i个选手在第j天所遇到的选手。其中1≤i≤n，1≤j≤n-1。8个选手的比赛日程表如下图：

* 分析

> 按分治策略，我们可以将所有的选手分为两半，则n个选手的比赛日程表可以通过n/2个选手的比赛日程表来决定。递归地用这种一分为二的策略对选手进行划分，直到只剩下两个选手时，比赛日程表的制定就变得很简单。这时只要让这两个选手进行比赛就可以了。如上图，所列出的正方形表是8个选手的比赛日程表。其中左上角与左下角的两小块分别为选手1至选手4和选手5至选手8前3天的比赛日程。据此，将左上角小块中的所有数字按其相对位置抄到右下角，又将左下角小块中的所有数字按其相对位置抄到右上角，这样我们就分别安排好了选手1至选手4和选手5至选手8在后4天的比赛日程。依此思想容易将这个比赛日程表推广到具有任意多个选手的情形。


* 代码

```
package MatchTable; /**
 * Created by Administrator on 2016/1/17.
 */
import java.util.Scanner;
public class MatchTable {
    public  void Table(int k, int n, int[][] a) {
        for(int i=1; i<= n; i++)
            a[1][i]=i;//设置日程表第一行

        int m = 1;//每次填充时，起始填充位置
        for(int s=1; s<=k; s++)
        {
            n /= 2;
            for(int t=1; t<=n; t++)
            {
                for(int i=m+1; i<=2*m; i++)//控制行
                {
                    for(int j=m+1; j<=2*m; j++)//控制列
                    {
                        a[i][j+(t-1)*m*2] = a[i-m][j+(t-1)*m*2-m];//右下角等于左上角的值
                        a[i][j+(t-1)*m*2-m] = a[i-m][j+(t-1)*m*2];//左下角等于右上角的值
                    }

                }
            }
            m *= 2;
        }

    }


    public static void main(String args[]) {
        System.out.println("请输入运动员的个数");
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        double x =Math.log(n)/Math.log(2);
        int k = (int)x;
        MatchTable t = new MatchTable();
        int[][] a = new int[n+1][n+1];
        t.Table(k,n,a);
        System.out.println(n + "名运动员的比赛日程表是：");
        for (int i = 1;i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                System.out.print(a[i][j] + " ");
            }
            System.out.println();
        }


    }
}
```
