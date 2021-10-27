---
title: hadoop & spark 分布式集群搭建
date: 2020-03-21
categories:
- 并行计算
tags:
- hadoop
- spark
keywords: hadoop,spark,分布式集群,搭建,ubuntu
description: 本文是自己用三台服务器搭建hadoop和spark分布式集群环境的过程记录。

---
使用三台主机搭建hadoop&spark完整教程
主要内容: 1)系统安装与配置,2)软件安装与配置,3)hadoop&spark安装与配置,4)集群启动&部署验证,5)集成阿里云,6)通过IDEA提交任务到spark
<!-- more -->

## 系统安装与配置
### 下载
https://ubuntu.com/download/server/thank-you?version=18.04.4&architecture=amd64

### 修改主机名
命令行修改
```
使用 hostname 修改当前主机名。
hostname new-hostname
```

修改/etc/sysconfig/network文件,将localhost.localdomain修改为指定hostname并保存文件退出
```
$ sudo vim /etc/sysconfig/network
NETWORKING=yes
HOSTNAME=localhost.localdomain
```

修改host
```
$ vi /etc/hosts
127.0.0.1 localhost localhost.localdomain localhost4 localhost4.localdomain4
::1 localhost localhost.localdomain localhost6 localhost6.localdomain6
将127.0.0.1 后指定的hosts改为新的hostname并保存文件退出
```

### 安装open-ssh
```
$ sudo apt update
$ sudo apt install openssh-server
$ sudo systemctl status ssh
$ sudo ufw allow ssh
```

### 创建用户
```
$ sudo useradd -m hadoop -s /bin/bash
$ sudo passwd hadoop
修改/etc/sudoder文件，给hadoop用户增加sudo权限。
```

### 修改Host
修改/etc/hosts文件，删除原来127.0.0.1到主机名的映射，增加如下配置。
* 前面是集群的IP，可以通过ip -a查看
* 后面是主机名
```
172.30.50.42    UbuntuMaster
172.30.50.81    UbuntuSlave1
172.30.50.84    UbuntuSlave2
```

### 配置免密码登陆
```
$ ssh-keygen -t rsa   #产生公钥与私钥对，执行三次回车
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
将～/.ssh目录下的id_rsa.pub,id_rsa,authorized_keys拷贝到其他两台server
```

## 软件安装与配置
### Java环境配置
下载Java JDK，放置到/opt目录下，并解压
``` shell
sudo mv jdk-8u241-linux-i586.tar.gz /opt
cd /opt
sudo tar -zxvf ./jdk-8u241-linux-i586.tar.gz
```

修改 /etc/profile文件，增加如下语句
``` shell
# java
export JAVA_HOME=/opt/jdk1.8.0_241
export CLASSPATH=:$JAVA_HOME/lib:$JAVA_HOME/jre/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JAVA_HOME/jre/bin:$PATH
```

刷新环境配置, 然后检测Java版本。
``` shell
source /etc/profile
java -version
如果报文件找不到，执行下面的语句
sudo apt-get install lib32stdc++6
```

### scala环境配置
下载scala，放置到/opt目录下，并解压
``` shell
wget https://downloads.lightbend.com/scala/2.12.10/scala-2.12.10.tgz
sudo mv ./scala-2.12.10.tgz /opt/
cd /opt/
sudo tar -zxf scala-2.12.10.tgz
```

修改环境变量,  vim /etc/profile，添加如下语句
``` shell
export SCALA_HOME=/opt/scala-2.12.10
export PATH=$PATH:$SCALA_HOME/bin
```

刷新环境配置, 然后检测Scala版本。
``` shell
source /etc/profile
scala -version
```

## hadoop & spark安装与配置
### hadoop的安装与配置
下载hadoop2.7，放置在/opt目录下，并解压
``` shell
$ wget https://archive.apache.org/dist/hadoop/core/hadoop-2.7.0/hadoop-2.7.0.tar.gz
$ tar -zxvf ./hadoop-2.7.0.tar.gz
$ sudo mv hadoop-2.7.0 /opt
```

修改环境变量，编辑/etc/profile文件，添加如下程序
``` shell
export HADOOP_HOME=/opt/hadoop-2.7.0
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_ROOT_LOGGER=INFO,console
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"
```

在hadoop-2.7.0目录下添加目录
``` shell
$ mkdir tmp
$ mkdir hdfs
$ mkdir hdfs/name
$ mkdir hdfs/data
```

修改$HADOOP_HOME/etc/hadoop/hadoop-env.sh，修改JAVA_HOME 如下：
```
export JAVA_HOME=/opt/jdk1.8.0_241
```

修改$HADOOP_HOME/etc/hadoop/slaves，将原来的localhost删除，添加如下内容：
```
UbuntuSlaver1
UbuntuSlaver2
```

修改$HADOOP_HOME/etc/hadoop/core-site.xml，修改为如下内容：
``` xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://UbuntuMaster:9000</value>
    </property>
    <property>
        <name>io.file.buffer.size</name>
        <value>131072</value>
    </property>
    <property>
        <name>hadoop.tmp.dir</name>
        <value>/opt/hadoop-2.7.0/tmp</value>
    </property>
</configuration>
```

修改$HADOOP_HOME/etc/hadoop/hdfs-site.xml
``` xml
<configuration>
    <property>
      <name>dfs.namenode.secondary.http-address</name>
      <value>UbuntuMaster:50090</value>
    </property>
    <property>
      <name>dfs.replication</name>
      <value>2</value>
    </property>
    <property>
      <name>dfs.namenode.name.dir</name>
      <value>file:/opt/hadoop-2.7.0/hdfs/name</value>
    </property>
    <property>
      <name>dfs.datanode.data.dir</name>
      <value>file:/opt/hadoop-2.7.0/hdfs/data</value>
    </property>
</configuration>
```

在$HADOOP_HOME/etc/hadoop目录下复制template，生成xml，命令如下：
``` xml
cp mapred-site.xml.template mapred-site.xml
修改$HADOOP_HOME/etc/hadoop/mapred-site.xml

<configuration>
    <property>
            <name>mapreduce.framework.name</name>
    <value>yarn</value>
    </property>
    <property>
            <name>mapreduce.jobhistory.address</name>
    <value>UbuntuMaster:10020</value>
    </property>
    <property>
    <name>mapreduce.jobhistory.address</name>
    <value>UbuntuMaster:19888</value>
    </property>
</configuration>
```

修改$HADOOP_HOME/etc/hadoop/yarn-site.xml
``` xml
<configuration>
     <property>
         <name>yarn.nodemanager.aux-services</name>
         <value>mapreduce_shuffle</value>
     </property>
     <property>
         <name>yarn.resourcemanager.address</name>
         <value>UbuntuMaster:8032</value>
     </property>
     <property>
         <name>yarn.resourcemanager.scheduler.address</name>
         <value>UbuntuMaster:8030</value>
     </property>
     <property>
         <name>yarn.resourcemanager.resource-tracker.address</name>
         <value>UbuntuMaster:8031</value>
     </property>
     <property>
         <name>yarn.resourcemanager.admin.address</name>
         <value>UbuntuMaster:8033</value>
     </property>
     <property>
         <name>yarn.resourcemanager.webapp.address</name>
         <value>UbuntuMaster:8088</value>
     </property>
</configuration>
```

### spark的安装与配置
下载hadoop2.7，放置在/opt目录下，并解压
``` shell
$ wget http://apache.communilink.net/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
$ tar -zxvf spark-2.4.5-bin-hadoop2.7.tgz
$ sudo mv spark-2.4.5-bin-hadoop2.7 /opt
```

修改/etc/profile，增加如下内容。
``` shell
export SPARK_HOME=/opt/spark-2.4.5-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
```

配置spark-env.sh文件
``` shell
$ cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh
在文件末尾添加如下内容：
export SCALA_HOME=/opt/scala-2.12.10
export JAVA_HOME=/opt/jdk1.8.0_241
export HADOOP_HOME=/opt/hadoop-2.7.0
export SPARK_WORKER_MEMORY=6g
export HADOOP_CONF_DIR=/opt/hadoop-2.7.0/etc/hadoop
export SPARK_MASTER_IP=172.30.50.42
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
```

配置slaves文件,添加如下内容
``` shell
cp $SPARK_HOME/conf/slaves.template $SPARK_HOME/conf/slaves
在文件末尾添加如下内容：
UbuntuMaster
UbuntuSlave1
UbuntuSlave2
```

### 同步配置&初始化集群
拷贝软件配置
``` shell
$ scp -r /opt/jdk1.8.0_241 hadoop@UbuntuSlave1:/opt
$ scp -r /opt/jdk1.8.0_241 hadoop@UbuntuSlave2:/opt
$ scp -r /opt/hadoop-2.7.0 hadoop@UbuntuSlave1:/opt
$ scp -r /opt/hadoop-2.7.0 hadoop@UbuntuSlave2:/opt
$ scp -r /opt/spark-2.4.5-bin-hadoop2.7 hadoop@UbuntuSlave1:/opt
$ scp -r /opt/spark-2.4.5-bin-hadoop2.7 hadoop@UbuntuSlave2:/opt
```

复制/etc/profile的配置到Slave
``` shell
# java
export JAVA_HOME=/opt/jdk1.8.0_241
export CLASSPATH=:$JAVA_HOME/lib:$JAVA_HOME/jre/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JAVA_HOME/jre/bin:$PATH

# scala
export SCALA_HOME=/opt/scala-2.12.10
export PATH=$PATH:$SCALA_HOME/bin

# hadoop
export HADOOP_HOME=/opt/hadoop-2.7.0
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME
export HADOOP_ROOT_LOGGER=INFO,console
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="-Djava.library.path=$HADOOP_HOME/lib/native"

# spark
export SPARK_HOME=/opt/spark-2.4.5-bin-hadoop2.7
export PATH=$PATH:$SPARK_HOME/bin
```

初始化Hadoop集群
``` shell
$ hadoop namenode -format
```

## 集群启动&部署验证
### hadoop集群启动
在Master节点，执行一下命令，启动集群。
``` shell
/opt/hadoop-2.7.0/sbin/start-all.sh
```

查看Hadoop是否启动成功，输入命令：jps
Master显示：SecondaryNameNode，ResourceManager，NameNode
Slaver显示：NodeManager，DataNode

管理界面
访问http://UbuntuMaster:50070, 查看 NameNode 和 Datanode 信息，还可以在线查看 HDFS 中的文件。

### hadoop集群验证

``` shell
cd  $HADOOP_HOME

bin/hadoop fs -rm -r /output
bin/hadoop fs -mkdir /input
bin/hadoop fs -put $HADOOP_HOME/README.txt /input
bin/hadoop fs -ls  /input
bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.0.jar wordcount  /input/README.txt  /output

bin/hadoop fs -ls  /output
bin/hadoop fs -cat /output/part-r-00000
```

### spark集群启动
在Master节点，执行一下命令，启动集群。
```
/opt/spark-2.4.5-bin-hadoop2.7/sbin/start-all.sh
```

查看Hadoop是否启动成功，输入命令：jps
Master显示：Master
Slaver显示：Worker

管理界面
访问http://UbuntuMaster:8080, 可以看到三个Worker

### spark集群验证
``` shell
$ spark-submit \
--class org.apache.spark.examples.SparkPi \
--master spark://UbuntuMaster:7077 \
--executor-memory 1G --total-executor-cores 2 \
/opt/spark-2.4.5-bin-hadoop2.7/examples/jars/spark-examples_2.11-2.4.5.jar \
100
```

## 集成阿里云
hadoop 2.9以后才支持oss的读写，我们使用的是2.7，需要自己配置。
下载支持包，并解压hadoop-aliyun-2.7.2.jar
http://gosspublic.alicdn.com/hadoop-spark/hadoop-oss-2.7.2.tar.gz

将文件hadoop-aliyun-2.7.2.jar复制到```$HADOOP_HOME/share/hadoop/tools/lib/```目录下

修改```$HADOOP_HOME/libexec/hadoop-config.sh```文件，再文件末尾增加```CLASSPATH=$CLASSPATH:$TOOL_PATH```

修改core-site.xml的配置
``` xml
    <property>
        <name>fs.oss.accessKeyId</name>
        <value>xxxx</value>
    </property>

    <property>
        <name>fs.oss.accessKeySecret</name>
        <value>xxx</value>
    </property>

    <property>
        <name>fs.oss.endpoint</name>
        <value>oss-us-east-1.aliyuncs.com</value>
    </property>

    <property>
        <name>fs.oss.impl</name>
        <value>org.apache.hadoop.fs.aliyun.oss.AliyunOSSFileSystem</value>
    </property>

    <property>
        <name>fs.oss.buffer.dir</name>
        <value>/tmp/oss</value>
    </property>

    <property>
        <name>fs.oss.connection.secure.enabled</name>
        <value>false</value>
    </property>

    <property>
        <name>fs.oss.connection.maximum</name>
        <value>2048</value>
    </property>
```

## 通过IDEA提交任务到spark
https://blog.csdn.net/yiluohan0307/article/details/80048765
