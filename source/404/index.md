---
title: '404 ' 
date: 2020-11-10 10:07:40
permalink: /404
---
<!DOCTYPE HTML>
<html>
<head>
  <meta http-equiv="content-type" content="text/html;charset=utf-8;"/>
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="robots" content="all" />
  <meta name="robots" content="index,follow"/>
</head>
<body>

<!-- markdownlint-disable MD039 MD033 -->

很抱歉，你目前访问的页面并不存在。

预计将在约 <span id="timeout">5</span> 秒后返回首页。

如果你很急着想看文章，你可以 **[点这里](https://imzhanghao.com/)** 返回首页。

<script>
let countTime = 5;

function count() {
  
  document.getElementById('timeout').textContent = countTime;
  countTime -= 1;
  if(countTime === 0){
    location.href = 'https://imzhanghao.com/';
  }
  setTimeout(() => {
    count();
  }, 1000);
}

count();
</script>

</body>
</html>
