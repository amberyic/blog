---
title: Hexo + Github Page搭建博客
---

## 准备环境
- Hexo是基于Node.js驱动的一款博客框架。

```
$ git version
$ node -v
$ npm -v
```

## 安装 Hexo
- 安装方法参考官网: https://hexo.io/zh-cn/

```
$ npm install hexo-cli -g
$ hexo init blog
$ cd blog
$ npm install
$ hexo server
```

## GitHub绑定
- 创建仓库名为abc.github.io的项目，其中abc为自己github的昵称
- 然后项目就建成了，点击Settings，向下拉到最后有个GitHub Pages，点击Choose a theme选择一个主题。
- 在博客根目录下的_config.yml文件最后增加下面的配置。

```
deploy:
  type: git
  repository: https://github.com/abc/abc.github.io
  branch: master
```

## 常用命令
```
$ hexo new post "article title"
$ hexo g
$ hexo s
$ hexo d
```

## 绑定域名
- 自己github域名的地址
```
$ ping abc.github.io
```
- 购买域名并将域名的A记录指到上一步得到的IP
- 在hexo目录下的source目录下添加CNAME，写上自己的域名。

## hexo备份
- clone博客的项目，创建hexo分支。
- 如果已经hexo d了，切换到hexo分支后，将内容从git中删除。
- 然后将hexo 文件夹中的_config.yml、themes/、source/、scaffolds/、package.json 和 .gitignore 复制至 abc.github.io 文件夹，并删除 themes/next/下的.git目录。将内容加入到代码仓库。
- 执行npm install 和 npm install hexo-deployer-git
- 执行 hexo g -d 生成静态网页部署至 Github 上

## hexo恢复
- 克隆博客的代码仓库
- 切换到hexo分支，执行以下命令

```
$ npm install hexo-cli -g
$ npm install
$ npm install hexo-deployer-git
```
