---
title: 五大经典算法|5.回溯法&分支界定法
date: 2020-05-10
categories:
- 五大经典算法
tags:
- 回溯法
- 分支界定法
---
> 这个部分在我们的课程中主要是在树及图的深度广度搜索部分有涉及，另外迷宫问题求解也有涉及。
> 经典例题:1)迷宫问题(maze problem),2) 01背包问题,3)八皇后问题,4)幂集,5)子集和问题
<!-- more -->

## 概念
* 回溯法
  - 回溯算法实际上一个类似枚举的搜索尝试过程，主要是在搜索尝试过程中寻找问题的解，当发现已不满足求解条件时，就“回溯”返回，尝试别的路径。
  - 回溯法是一种选优搜索法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法，而满足回溯条件的某个状态的点称为“回溯点”。
  - 许多复杂的，规模较大的问题都可以使用回溯法，有“通用解题方法”的美称。

* 分支界定法
  - 类似于回溯法，也是一种在问题的解空间树T上搜索问题解的算法。但在一般情况下，分支限界法与回溯法的求解目标不同。回溯法的求解目标是找出T中满足约束条件的所有解，而分支限界法的求解目标则是找出满足约束条件的一个解，或是在满足约束条件的解中找出使某一目标函数值达到极大或极小的解，即在某种意义下的最优解。


## 思想

* 回溯法
  - 在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根结点出发深度探索解空间树。当探索到某一结点时，要先判断该结点是否包含问题的解，如果包含，就从该结点出发继续探索下去，如果该结点不包含问题的解，则逐层向其祖先结点回溯。(其实回溯法就是对隐式图的深度优先搜索算法)。
  - 若用回溯法求问题的所有解时，要回溯到根，且根结点的所有可行的子树都要已被搜索遍才结束。
  - 而若使用回溯法求任一个解时，只要搜索到问题的一个解就可以结束。


* 分支界定法
  - 1 分支搜索算法
  > 所谓“分支”就是采用广度优先的策略，依次搜索E-结点的所有分支，也就是所有相邻结点，抛弃不满足约束条件的结点，其余结点加入活结点表。然后从表中选择一个结点作为下一个E-结点，继续搜索。
  > 选择下一个E-结点的方式不同，则会有几种不同的分支搜索方式。
    - 1)FIFO搜索
    - 2)LIFO搜索
    - 3)优先队列式搜索
  - (2)分支限界搜索算法



## 题目
### 1.迷宫问题(maze problem)
* [题目](https://blog.csdn.net/K346K346/article/details/51289478)

> 给定一个迷宫，指明起点和终点，找出从起点出发到终点的有效可行路径，就是迷宫问题(maze problem)
>
> 迷宫可以以二维数组来存储表示。0表示通路，1表示障碍。注意这里规定移动可以从上、下、左、右四方方向移动。坐标以行和列表示，均从0开始，给定起点(0,0)和终点(4,4)，迷宫表示如下：
>
> int maze[5][5]={
>     {0,0,0,0,0},
>     {0,1,0,1,0},
>     {0,1,1,0,0},
>     {0,1,1,0,1},
>     {0,0,0,0,0}
> };

* 分析

> 那么下面的迷宫就有两条可行的路径，分别为：
> (1)(0,0) (0,1) (0,2) (0,3) (0,4) (1,4) (2,4) (2,3) (3,3) (4,3) (4,4)；
> (2)(0,0) (1,0) (2,0) (3,0) (4,0) (4,1) (4,2) (4,3) (4,4) ；
>
> 可见，迷宫可行路径有可能是多条，且路径长度可能不一。
>
> 迷宫问题的求解可以抽象为连通图的遍历，因此主要有两种方法。
>
> 第一种方法是：深度优先搜索(DFS)加回溯。
>
> 其优点：无需像广度优先搜索那样(BFS)记录前驱结点。
> 其缺点：找到的第一条可行路径不一定是最短路径，如果需要找到最短路径，那么需要找出所有可行路径后，再逐一比较，求出最短路径。
>
> 第二种方法是：广度优先搜索(BFS)。
> 其优点：找出的第一条路径就是最短路径。
> 其缺点：需要记录结点的前驱结点，来形成路径。


* 代码

> 方法一： 深度优先搜索(DFS)加回溯求解第一条可行路径

```
实现步骤
(1)给定起点和终点，判断二者的合法性，如果不合法，返回；
(2)如果起点和终点合法，将起点入栈；
(3)取栈顶元素，求其邻接的未被访问的无障碍结点。求如果有，记其为已访问，并入栈。
   如果没有则回溯上一结点，具体做法是将当前栈顶元素出栈。
   其中，求邻接无障碍结点的顺序可任意，本文实现是以上、右、下、左的顺序求解。
(4)重复步骤(3)，直到栈空(没有找到可行路径)或者栈顶元素等于终点(找到第一条可行路径)

#include <iostream>
#include <stack>
using namespace std;

struct Point{
    //行与列
    int row;
    int col;
    Point(int x,int y){
        this->row=x;
        this->col=y;
    }

    bool operator!=(const Point& rhs){
        if (this->row!=rhs.row||this->col!=rhs.col)
            return true;
        return false;
    }
};

//func:获取相邻未被访问的节点
//para:mark:结点标记，point：结点，m：行，n：列
//ret:邻接未被访问的结点
Point getAdjacentNotVisitedNode(bool** mark,Point point,int m,int n){
    Point resP(-1,-1);
    if (point.row-1>=0&&mark[point.row-1][point.col]==false){//上节点满足条件
        resP.row=point.row-1;
        resP.col=point.col;
        return resP;
    }
    if (point.col+1<n&&mark[point.row][point.col+1]==false){//右节点满足条件
        resP.row=point.row;
        resP.col=point.col+1;
        return resP;
    }
    if (point.row+1<m&&mark[point.row+1][point.col]==false){//下节点满足条件
        resP.row=point.row+1;
        resP.col=point.col;
        return resP;
    }
    if (point.col-1>=0&&mark[point.row][point.col-1]==false){//左节点满足条件
        resP.row=point.row;
        resP.col=point.col-1;
        return resP;
    }
    return resP;
}


//func：给定二维迷宫，求可行路径
//para:maze：迷宫；m：行；n：列；startP：开始结点 endP：结束结点； pointStack：栈，存放路径结点
//ret:无
void mazePath(void* maze,int m,int n,const Point& startP,Point endP,stack<Point>& pointStack){
    //将给定的任意列数的二维数组还原为指针数组，以支持下标操作
    int** maze2d=new int*[m];
    for (int i=0;i<m;++i){
        maze2d[i]=(int*)maze+i*n;
    }

    if (maze2d[startP.row][startP.col]==1||maze2d[endP.row][endP.col]==1)
        return ;                    //输入错误

    //建立各个节点访问标记
    bool** mark=new bool*[m];
    for (int i=0;i<m;++i){
        mark[i]=new bool[n];
    }
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            mark[i][j]=*((int*)maze+i*n+j);
        }
    }

    //将起点入栈
    pointStack.push(startP);
    mark[startP.row][startP.col]=true;

    //栈不空并且栈顶元素不为结束节点
    while(pointStack.empty()==false&&pointStack.top()!=endP){
        Point adjacentNotVisitedNode=getAdjacentNotVisitedNode(mark,pointStack.top(),m,n);
        if (adjacentNotVisitedNode.row==-1){ //没有未被访问的相邻节点
            pointStack.pop(); //回溯到上一个节点
            continue;
        }

        //入栈并设置访问标志为true
        mark[adjacentNotVisitedNode.row][adjacentNotVisitedNode.col]=true;
        pointStack.push(adjacentNotVisitedNode);
    }
}

int main(){
    int maze[5][5]={
        {0,0,0,0,0},
        {0,1,0,1,0},
        {0,1,1,0,0},
        {0,1,1,0,1},
        {0,0,0,0,0}
    };

    Point startP(0,0);
    Point endP(4,4);
    stack<Point>  pointStack;
    mazePath(maze,5,5,startP,endP,pointStack);

    //没有找打可行解
    if (pointStack.empty()==true)
        cout<<"no right path"<<endl;
    else{
        stack<Point> tmpStack;
        cout<<"path:";
        while(pointStack.empty()==false){
            tmpStack.push(pointStack.top());
            pointStack.pop();
        }
        while (tmpStack.empty()==false){
            printf("(%d,%d) ",tmpStack.top().row,tmpStack.top().col);
            tmpStack.pop();
        }
    }
    getchar();
}

程序输出：path:(0,0) (0,1) (0,2) (0,3) (0,4) (1,4) (2,4) (2,3) (3,3) (4,3) (4,4)。

可见该条路径不是最短路径。因为程序中给定的迷宫还有一条更短路径为：(0,0) (1,0) (2,0) (3,0) (4,0) (4,1) (4,2) (4,3) (4,4) ；
```

> 方法二：改进深度优先搜索(DFS)加回溯求解最短路径

```
实现方法
根据上面的方法我们可以在此基础之上进行改进，求出迷宫的最短的路径。具体做法如下：
(1)让已经访问过的结点可以再次被访问，具体做法是将mark标记改为当前结点到起点的距离，作为当前结点的权值。即从起点开始出发，向四个方向查找，每走一步，把走过的点的值+1；
(2)寻找栈顶元素的下一个可访问的相邻结点，条件就是栈顶元素的权值加1必须小于下一个节点的权值(墙不能走，未被访问的结点权值为0)；
(3)如果访问到终点，记录当前最短的路径。如果不是，则继续寻找下一个结点；
(4)重复步骤(2)和(3)直到栈空(迷宫中所有符合条件的结点均被访问)。

#include <iostream>
#include <stack>
#include <vector>
using namespace std;

struct Point{
    //行与列
    int row;
    int col;
    Point(int x,int y){
        this->row=x;
        this->col=y;
    }

    bool operator!=(const Point& rhs){
        if (this->row!=rhs.row||this->col!=rhs.col)
            return true;
        return false;
    }

    bool operator==(const Point& rhs) const{
        if (this->row==rhs.row&&this->col==rhs.col)
            return true;
        return false;
    }
};

int maze[5][5]={
    {0, 0, 0, 0,0},
    {0,-1, 0,-1,0},
    {0,-1,-1, 0,0},
    {0,-1,-1, 0,-1},
    {0, 0, 0, 0, 0}
};

//func:获取相邻未被访问的节点
//para:mark:结点标记；point：结点；m：行；n：列;endP:终点
//ret:邻接未被访问的结点
Point getAdjacentNotVisitedNode(int** mark,Point point,int m,int n,Point endP){
    Point resP(-1,-1);
    if (point.row-1>=0){
        if (mark[point.row-1][point.col]==0||mark[point.row][point.col]+1<mark[point.row-1][point.col]){//上节点满足条件
            resP.row=point.row-1;
            resP.col=point.col;
            return resP;
        }
    }
    if (point.col+1<n){
        if (mark[point.row][point.col+1]==0||mark[point.row][point.col]+1<mark[point.row][point.col+1]){//右节点满足条件
            resP.row=point.row;
            resP.col=point.col+1;
            return resP;
        }
    }
    if (point.row+1<m){
        if (mark[point.row+1][point.col]==0||mark[point.row][point.col]+1<mark[point.row+1][point.col]){//下节点满足条件
            resP.row=point.row+1;
            resP.col=point.col;
            return resP;
        }
    }
    if (point.col-1>=0){
        if (mark[point.row][point.col-1]==0||mark[point.row][point.col]+1<mark[point.row][point.col-1]){//左节点满足条件
            resP.row=point.row;
            resP.col=point.col-1;
            return resP;
        }
    }
    return resP;
}

//func：给定二维迷宫，求可行路径
//para:maze：迷宫；m：行；n：列；startP：开始结点 endP：结束结点； pointStack：栈，存放路径结点;vecPath:存放最短路径
//ret:无
void mazePath(void* maze,int m,int n, Point& startP, Point endP,stack<Point>& pointStack,vector<Point>& vecPath){
    //将给定的任意列数的二维数组还原为指针数组，以支持下标操作
    int** maze2d=new int*[m];
    for (int i=0;i<m;++i){
        maze2d[i]=(int*)maze+i*n;
    }

    if (maze2d[startP.row][startP.col]==-1||maze2d[endP.row][endP.col]==-1)
        return ;                    //输入错误

    //建立各个节点访问标记，表示结点到到起点的权值，也记录了起点到当前结点路径的长度
    int** mark=new int*[m];
    for (int i=0;i<m;++i){
        mark[i]=new int[n];
    }
    for (int i=0;i<m;++i){
        for (int j=0;j<n;++j){
            mark[i][j]=*((int*)maze+i*n+j);
        }
    }
    if (startP==endP){//起点等于终点
        vecPath.push_back(startP);
        return;
    }

    //增加一个终点的已被访问的前驱结点集
    vector<Point> visitedEndPointPreNodeVec;

    //将起点入栈
    pointStack.push(startP);
    mark[startP.row][startP.col]=true;

    //栈不空并且栈顶元素不为结束节点
    while(pointStack.empty()==false){
        Point adjacentNotVisitedNode=getAdjacentNotVisitedNode(mark,pointStack.top(),m,n,endP);
        if (adjacentNotVisitedNode.row==-1){ //没有符合条件的相邻节点
            pointStack.pop(); //回溯到上一个节点
            continue;
        }
        if (adjacentNotVisitedNode==endP){//以较短的路劲，找到了终点,
            mark[adjacentNotVisitedNode.row][adjacentNotVisitedNode.col]=mark[pointStack.top().row][pointStack.top().col]+1;
            pointStack.push(endP);
            stack<Point> pointStackTemp=pointStack;
            vecPath.clear();
            while (pointStackTemp.empty()==false){
                vecPath.push_back(pointStackTemp.top());//这里vecPath存放的是逆序路径
                pointStackTemp.pop();
            }
            pointStack.pop(); //将终点出栈

            continue;
        }
        //入栈并设置访问标志为true
        mark[adjacentNotVisitedNode.row][adjacentNotVisitedNode.col]=mark[pointStack.top().row][pointStack.top().col]+1;
        pointStack.push(adjacentNotVisitedNode);
    }
}

int main(){
    Point startP(0,0);
    Point endP(4,4);
    stack<Point>  pointStack;
    vector<Point> vecPath;
    mazePath(maze,5,5,startP,endP,pointStack,vecPath);

    if (vecPath.empty()==true)
        cout<<"no right path"<<endl;
    else{
        cout<<"shortest path:";
        for (auto i=vecPath.rbegin();i!=vecPath.rend();++i)
            printf("(%d,%d) ",i->row,i->col);
    }

    getchar();
}

```

> 方法3: 广度优先搜索(BFS)求解迷宫的最短路径

```
广度优先搜索的优点是找出的第一条路径就是最短路径，所以经常用来搜索最短路径，思路和图的广度优先遍历一样，需要借助于队列。
具体步骤：
(1)从入口元素开始，判断它上下左右的邻边元素是否满足条件，如果满足条件就入队列；
(2)取队首元素并出队列。寻找其相邻未被访问的元素，将其如队列并标记元素的前驱节点为队首元素。
(3)重复步骤(2)，直到队列为空(没有找到可行路径)或者找到了终点。最后从终点开始，根据节点的前驱节点找出一条最短的可行路径。

#include <iostream>
#include <queue>
using namespace std;

struct Point{
    //行与列
    int row;
    int col;

    //默认构造函数
    Point(){
        row=col=-1;
    }

    Point(int x,int y){
        this->row=x;
        this->col=y;
    }

    bool operator==(const Point& rhs) const{
        if (this->row==rhs.row&&this->col==rhs.col)
            return true;
        return false;
    }
};

int maze[5][5]={
    {0,0,0,0,0},
    {0,1,0,1,0},
    {0,1,1,1,0},
    {0,1,0,0,1},
    {0,0,0,0,0}
};

void mazePath(void* maze,int m,int n, Point& startP, Point endP,vector<Point>& shortestPath){
    int** maze2d=new int*[m];
    for (int i=0;i<m;++i){
        maze2d[i]=(int*)maze+i*n;
    }

    if (maze2d[startP.row][startP.col]==1||maze2d[startP.row][startP.col]==1) return ; //输入错误

    if (startP==endP){ //起点即终点
        shortestPath.push_back(startP);
        return;
    }

    //mark标记每一个节点的前驱节点，如果没有则为(-1，-1)，如果有，则表示已经被访问
    Point** mark=new Point*[m];
    for (int i=0;i<m;++i){
        mark[i]=new Point[n];
    }

    queue<Point> queuePoint;
    queuePoint.push(startP);
    //将起点的前驱节点设置为自己
    mark[startP.row][startP.col]=startP;

    while(queuePoint.empty()==false){
        Point pointFront=queuePoint.front();
        queuePoint.pop();

        if (pointFront.row-1>=0 && maze2d[pointFront.row-1][pointFront.col]==0){//上节点连通
            if (mark[pointFront.row-1][pointFront.col]==Point()){//上节点未被访问，满足条件，如队列
                mark[pointFront.row-1][pointFront.col]=pointFront;
                queuePoint.push(Point(pointFront.row-1,pointFront.col)); //入栈
                if (Point(pointFront.row-1,pointFront.col)==endP){ //找到终点
                    break;
                }
            }
        }

        if (pointFront.col+1<n && maze2d[pointFront.row][pointFront.col+1]==0){//右节点连通
            if (mark[pointFront.row][pointFront.col+1]==Point()){//右节点未被访问，满足条件，如队列
                mark[pointFront.row][pointFront.col+1]=pointFront;
                queuePoint.push(Point(pointFront.row,pointFront.col+1));    //入栈
                if (Point(pointFront.row,pointFront.col+1)==endP){ //找到终点
                    break;
                }
            }
        }

        if (pointFront.row+1<m && maze2d[pointFront.row+1][pointFront.col]==0){//下节点连通
            if (mark[pointFront.row+1][pointFront.col]==Point()){//下节点未被访问，满足条件，如队列
                mark[pointFront.row+1][pointFront.col]=pointFront;
                queuePoint.push(Point(pointFront.row+1,pointFront.col));    //入栈
                if (Point(pointFront.row+1,pointFront.col)==endP){ //找到终点
                    break;
                }
            }
        }

        if (pointFront.col-1>=0 && maze2d[pointFront.row][pointFront.col-1]==0){//左节点连通
            if (mark[pointFront.row][pointFront.col-1]==Point()){//上节点未被访问，满足条件，如队列
                mark[pointFront.row][pointFront.col-1]=pointFront;
                queuePoint.push(Point(pointFront.row,pointFront.col-1));    //入栈
                if (Point(pointFront.row,pointFront.col-1)==endP){ //找到终点
                    break;
                }
            }
        }
    }
    if (queuePoint.empty()==false){
        int row=endP.row;
        int col=endP.col;
        shortestPath.push_back(endP);
        while(!(mark[row][col]==startP)){
            shortestPath.push_back(mark[row][col]);
            row=mark[row][col].row;
            col=mark[row][col].col;
        }
        shortestPath.push_back(startP);
    }
}

int main(){
    Point startP(0,0);
    Point endP(4,4);
    vector<Point> vecPath;
    mazePath(maze,5,5,startP,endP,vecPath);

    if (vecPath.empty()==true)
        cout<<"no right path"<<endl;
    else{
        cout<<"shortest path:";
        for (auto i=vecPath.rbegin();i!=vecPath.rend();++i)
            printf("(%d,%d) ",i->row,i->col);
    }

    getchar();
}

```


### 2. 01背包问题
* [题目](http://fuliang.iteye.com/blog/165308)

 > 给定N中物品和一个背包。物品i的重量是Wi,其价值位Vi ，背包的容量为C。问应该如何选择装入背包的物品，使得转入背包的物品的总价值为最大？？


* 分析
> 0-1背包是子集合选取问题,一般情况下0-1背包是个NP问题.
> 第一步　确定解空间：装入哪几种物品.
> 第二步　确定易于搜索的解空间结构：
> 可以用数组p,w分别表示各个物品价值和重量。
> 用数组x记录，是否选种物品.
> 第三步　以深度优先的方式搜索解空间，并在搜索的过程中剪枝
> 我们同样可以使用子集合问题的框架来写我们的代码，和前面子集和数问题相差无几。

* 代码

```
#include<iostream>
#include<algorithm>
using namespace std;

class Knapsack{
public:
    Knapsack(double *pp,double *ww,int nn,double cc){
       p = pp;
       w = ww;
       n = nn;
       c = cc;
       cw = 0;
       cp = 0;
       bestp = 0;
       x = new int[n];
       cx = new int[n];
    }

    void knapsack(){
       backtrack(0);
     }

    void backtrack(int i){//回溯法
        if (i > n){
            if (cp > bestp){
               bestp = cp;
               for (int i = 0; i < n; i++)
             x[i] = cx[i];
            }
            return;
        }

        if (cw + w[i] <= c){//搜索右子树
          cw += w[i];
          cp += p[i];
          cx[i] = 1;
          backtrack(i+1);
          cw -= w[i];
          cp -= p[i];
        }
        cx[i] = 0;
        backtrack(i+1);//搜索左子树
    }

    void printResult(){
       cout << "可以装入的最大价值为:" << bestp << endl;
       cout << "装入的物品依次为:";
       for (int i = 0; i < n; i++){
         if (x[i] == 1)
             cout << i+1 << " ";
       }
       cout << endl;
    }

private:
   double *p,*w;
   int n;
   double c;
   double bestp,cp,cw;//最大价值，当前价值，当前重量
   int *x,*cx;
};

int main(){
　　double p[4] = {9,10,7,4},w[4] = {3,5,2,1};
    Knapsack ks = Knapsack(p,w,4,7);
    ks.knapsack();
　　ks.printResult();
　　return 0;
}
```


### 3.八皇后问题
* [题目](http://fuliang.iteye.com/blog/164744)

> 八皇后问题是一个古老而著名的问题，是回溯算法的典型例题。该问题是十九世纪著名的数学家高斯1850年提出：在8X8格的国际象棋上摆放八个皇后，使其不能互相攻击，即任意两个皇后都不能处于同一行、同一列或同一斜线上.

* 分析

> 第一步 定义问题的解空间
> 这个问题解空间就是8个皇后在棋盘中的位置.
> 第二步 定义解空间的结构
> 可以使用8*8的数组，但由于任意两个皇后都不能在同行，我们可以用数组下标表示
> 行，数组的值来表示皇后放的列，故可以简化为一个以维数组x[9]。
> 第三步 以深度优先的方式搜索解空间，并在搜索过程使用剪枝函数来剪枝
> 根据条件:x[i] == x[k]判断处于同一列
>          abs(k-i) == abs(x[k]-x[i]判断是否处于同一斜线
> 我们很容易写出剪枝函数：


* 代码

```
#include<iostream>
#include<cmath>
using namespace std;

int x[9];
void print(){
    for (int i = 1; i <= 8; i++)
           cout << x[i] << " ";
    cout << endl;
}

bool canPlace(int k){
    for (int i = 1; i < k; i++){
            //判断处于同一列或同一斜线
       if (x[i] == x[k] || abs(k-i) == abs(x[k]-x[i]))
           return false;
    }
    return true;
}

void queen(int i){
    if (i > 8){
        print();
        return;
    }
    for (int j = 1; j <= 8; j++){
      x[i] = j;
      if (canPlace(i)) queen(i+1);
    }
}

int main(){
  queen(1);
  return 0;
}
```

### 4.幂集
* 题目

> 幂集的每个元素是一个集合或者是一个空集。拿集合{A, B, C}来举例，这个集合的幂集为{ {A, B, C}, {A , B}, {A , C}, {B, C},{A}, {B}, {C}, {}}。可以看出分为3中状态:
>
> 1.空集
> 2.是集合中的一个元素组成的集合
> 3.是集合中的任意两个元素组成的集合
> 4.是集合中的三个元素组成的集合，就是它本身


* 分析

> 算法思想，集合中每个元素有两种状态，在幂集元素的集合中，不在集合中。可以用一颗二叉树形象的表示回溯遍历的过程


* 代码

```
#include <iostream>
using namespace std;
char *result;
char *element;
void OutputPowerSet(int len){ //输出幂集中的元素
  cout<<"{ ";
  int eln = 0;
  for (int i = 0; i < len; i++){
    if (result[i] != 0)
    {
      if (eln > 0)
        cout<<", "<<result[i];
      else
        cout<<result[i];
      eln++;
    }
  }
  cout<<" }; ";
}
void PowerSet(int k,int n){
  if (k > n)
  {
    OutputPowerSet(n);
  }else{
    result[k-1] = element[k-1]; //元素在幂集元素集合中
    PowerSet(k+1,n);
    result[k-1] = 0;//元素不在幂集元素集合中
    PowerSet(k+1,n);
  }
}
int main(){
  int num;
  cin>>num;    //输出要求幂集的初始集合元素个数
  element = new char[num];
  result = new char[num];
  int index = 0;
  while(index < num){
    cin>>element[index];  //输入集合元素，这里用字符代替
    index++;
  }
  PowerSet(1,num);
}
```

### 5.子集和问题
* 题目

> 存在S={x1,x2,..xn}.是一个正整数的集合，c是一个正整数。子集合问题判定是否存在一个子集S1(S1为S的子集)，使得该子集的和为c.
> 例子：S={1,3,8,9},C=9,则解为:s1={1,8},s2={9}
>
> 可以看出此算法的解空间为子集树，所以利用前面讲的模板，可以得到哦以下程序


* 分析


* 代码

```
/*
   名称：算法5-1
   问题重述：
      子集合问题,存在S={x1,x2,..xn}.是一个正整数的集合，c是一个正整数。子集合问题判定是否存在一个子集S1，使得其中一个子集的和为c.
  时间:2013/5/12
  作者：刘荣
*/

#include<stdio.h>
bool next(int a[],int n, int i, int s, int r, int c, int bextx[], int x[])
{
  int j;
  if (i >= n) {//到达叶子结点
    if (s == c) { //找到一个子集
      for (int k=0;k<n;k++) {//记录下子集
        bextx[k] = x[k];
      }
      return true;
    } else {//没有找到符合的子集
      return false;
    }
  }
  if (s >c || s+r <c) {
    return false;
  }
  x[i] = 1;
  if (next(a, n, i+1, s+a[i], r-a[i], c, bextx, x)) {
    return true;
  }
  x[i] = 0;
  return next(a, n, i+1, s, r-a[i], c, bextx, x);
}

bool solve(int a[],int n,int c,int bextx[]) {
  //int *bextx = new int[n];
  int *x = new int[n];
  int r = 0;
  for (int i=0; i<n; i++) {
    r += a[i];
  }
  return  next(a, n, 0, 0, r, c, bextx, x);
}

int main() {
  int a[]={1,2,6,8};
  int n=4;
  int c=8;
  int *bextx = new int[n];
  if (solve(a,n,c,bextx)) {
    printf("找到子集： \n\r");
    for (int i=0;i<n;i++) {
      printf("%d ",bextx[i]);
    }
  } else {
    printf("没有子集");
  }


  return 0;
}

```
