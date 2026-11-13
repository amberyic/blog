# v2grad 域名迁移的 Nginx 配置与回退步骤

更新：2026-11-13。状态：已获用户明确授权并完成生产切换与验收，Google 根域名及 www 地址更改均提交成功，两平台 sitemap 均处理成功，详见迁移计划中的上线记录。

## 2026-11-13 实际上线记录

- 用户明确回复“按方案上线吧”，授权安装证书、部署新站、DNS 与 301 切换及搜索平台迁移提交。
- 服务器配置与准备时的快照 SHA-256 一致；先备份至 `/root/backups/v2grad-20261113-1505/`，包含原 `default`、`nginx.conf`、启用链接清单、Git 钩子和 `old-site.tar.gz`。备份目录权限 700，旧站压缩包权限 600。原 `/var/www/hexo` 完整保留。
- 新旧证书安装至本文指定的 `/etc/nginx/ssl/` 独立目录，私钥均为 600，公钥匹配核对通过。原证书文件保留。
- 发布目录为 `/var/www/v2grad/releases/20261113-1505/`，`current` 链接已指向该版本；220 个文件共 13,021,641 字节，上传后逐文件 SHA-256 与本地隔离构建产物一致。网站文件 644，目录 755，产物没有 SSL 私钥或 ssl 目录。
- 新站先通过真实证书的 `curl --resolve` 验证，127/127 sitemap 路径返回 200，再添加 DNS：根域 A → `47.105.106.15`，www CNAME → `v2grad.com`，TTL 均为 600 秒；原 Google、Bing 和签证书验证记录均保留，未新增 AAAA。
- 15:20 核对发现旧根域名的 4 条境外 A 仍指向 GitHub Pages，导致 Google 获取旧站；已暂停四条记录并保留，境外访问回落到原默认 A `47.105.106.15`。原 www、oss CDN 及验证记录未改。旧值和状态保存在本机证据 `old-overseas-dns-before.json` 中。回退若需恢复境外 GitHub 线路，可重新启用这四条原记录，但须先评估 Google 迁移状态。
- 北京时间 15:08:28 已将本目录 `imzhanghao.conf` 安装为生效 `default`，`v2grad.conf` 单独启用；两次配置检查和 reload 均通过，Nginx 为 active。
- Google 已确认根域名与 www 的地址更改；若后续执行站点回退，须同时评估并按官方流程取消两项迁移，不能只恢复 Nginx 就视为完整反向迁移。
- 回退配置已放在备份目录中的 `imzhanghao.rollback.conf`，没有启用。回退时按下文步骤恢复旧站服务，保持新站可用，避免反向 301 循环。
- 原 Git 钩子未改，GitHub 未发布，未运行 `npm run deploy`。后续更新新站须发布到新的 releases 目录并原子切换 current；旧部署命令仍写旧目录，不能直接沿用作为新站发布方式。

下文的“候选”“待执行”“未上线”描述保留为准备阶段说明；实际执行状态以上述记录和迁移计划最新进度为准。

2026-11-12 补充：用户已提供新站证书，完成本地核对，尚未安装。证书覆盖 `v2grad.com` 与 `www.v2grad.com`，签发者为 DigiCert Encryption Everywhere DV TLS CA - G2，有效期为北京时间 2026-11-12 08:00:00 至 2027-02-10 07:59:59。PEM 包含叶证书与中间证书两张，使用本机信任库验证通过，证书公钥与所提供私钥匹配。

来源目录为 `/Users/zhanghao/Documents/28467091_www.v2grad.com_nginx`。获准上线后，将其中 `www.v2grad.com.pem` 安装为配置使用的 `fullchain.pem`，将 `www.v2grad.com.key` 安装为 `privkey.pem`；当前未复制到服务器。仓库根目录另有同内容未跟踪副本，包含私钥，不应加入 Git；文件保留，未改权限或删除。现有本地私钥权限为 664，正式安装按下文的 600 设置。

本目录配置不会进入 Hexo 静态产物。正式安装、申请证书、发布文件、修改解析、reload 或搜索回迁均须另行获得明确授权。不要把本目录全部复制到 sites-enabled；迁移与回退配置只能二选一。

## 现网检查结果

| 项目 | 实际情况 |
| --- | --- |
| 服务器 | 47.105.106.15，SSH 可访问 |
| Nginx | Ubuntu 的 nginx/1.14.0，主配置 /etc/nginx/nginx.conf |
| 生效站点 | /etc/nginx/sites-enabled/default 链接到 /etc/nginx/sites-available/default；conf.d 为空 |
| 原静态目录 | /var/www/hexo，223 个文件，约 13 MB |
| 原部署仓库 | /var/repo/blog.git，HEAD 指向 master |
| 接收钩子 | post-receive 执行 git checkout -f 到 /var/www/hexo，直接覆盖在用目录 |
| 本地 deploy | 同时配置 GitHub master 与上述服务器仓库；本次迁移不可直接运行 npm run deploy |
| 原证书 | /var/www/hexo/ssl/imzhanghao.com.pem，私钥为同目录 .key；仅查看公有证书元信息，未读取或下载私钥 |
| 覆盖域名 | imzhanghao.com、www.imzhanghao.com |
| 证书到期 | 2027-02-01 23:59:59 UTC，即北京时间 2027-02-02 07:59:59 |
| 自动续期 | 未发现 certbot、/etc/letsencrypt、/root/.acme.sh 或相关 systemd timer；已检查 root crontab、cron.d、cron.daily，未找到续期线索。不能据此排除阿里云侧或人工续签 |
| 其他原有路径 | /report.html 映射到 /data/www/goaccess/html/report.html |
| 磁盘 | /var/www 所在分区可用约 7.2 GB，可容纳独立站点和多份小型静态版本 |

证书位于网站 root 内，现有普通静态规则没有单独阻止 /ssl/。本次未请求线上密钥 URL，未判断是否发生外部获取。候选配置将证书移到独立目录，并让 /ssl/ 返回 404；原文件保留，证书轮换及线上影响调查不在本次只读检查范围。

## 候选文件的用途

- `v2grad.conf`：新站；HTTP 和 HTTPS www 入口直接 301 到 HTTPS 非 www，对应文件正常提供，缺失文件返回真实 404。
- `imzhanghao.conf`：切换时替换原 default；普通旧主站请求通过 `https://v2grad.com$request_uri` 逐页跳转，保留编码路径及查询参数。
- `imzhanghao.rollback.conf`：回退时替换 default，恢复原站静态目录和原报告，继续使用独立的证书路径。

旧站 robots.txt 保持 200 并允许抓取；ACME HTTP 验证路径预留独立 webroot；/report.html 保留原域名供运维使用，不属于博客迁移内容。图片域名 oss.imzhanghao.com 的 DNS/CDN 不变，配置没有将它列入 server_name。本机默认虚拟主机仍由旧站承接未知 Host；现有图片域名走 CDN，这次不修改该链路。

301 初始附带 `Cache-Control: max-age=300`，减少浏览器长期缓存造成的回退困难；不能保证客户端一定没有缓存。稳定后可另行调整，不把 301 改成 302。

## 目录与证书方案

建议沿用现有服务器，新增如下路径：

```text
/var/www/hexo                         原站保留，原 Git 钩子暂不改
/var/www/v2grad/releases/<版本时间>   每次放入一份全新静态产物
/var/www/v2grad/current              符号链接，指向正在使用的版本
/etc/nginx/ssl/imzhanghao.com/        旧站 fullchain.pem 与 privkey.pem
/etc/nginx/ssl/v2grad.com/            新站 fullchain.pem 与 privkey.pem
/var/www/acme-challenge/             预留 HTTP-01 验证目录
```

新证书必须同时覆盖 v2grad.com 和 www.v2grad.com。优先通过阿里云证书控制台做 DNS 验证签发，不依赖提前开放网站。若没有免费额度，先让用户选择费用，不自行购买。本次未申请证书、创建上述服务器路径或搬动原证书。

原证书可在服务器内复制至独立目录；私钥目录 700、私钥 600，公有证书 644。复制后核对证书链、SAN、有效期和公钥匹配，不能只凭文件名判断证书正确。旧证书必须持续有效，旧 HTTPS 请求先完成 TLS 握手才能收到 301。

目前没有证据证明自动续期可用，现阶段按人工续签管理；新旧证书签发或续签后均需重新核对有效期、替换对应独立目录并检查 reload。若改用 ACME DNS 自动续期，客户端安装、DNS 权限和定时任务单独授权；Certbot 的 manual 模式本身不等于无人值守续期。

## 获准切换后的执行顺序

以下是待执行步骤，不是本次执行记录。

1. 确认新域名备案已通过；新证书已签发且覆盖主域名和 www。确认候选文件与现网检查时的 default、Git 钩子没有未解释差异。
2. 在服务器创建带时间的备份目录，保存 default、nginx.conf、站点启用链接清单及 post-receive。保留 /var/www/hexo，不使用 git reset、旧钩子发布或覆盖旧站文件。
3. 在服务器内将旧证书复制到独立证书目录，安装新证书；保留原证书文件。核对权限、链、SAN、有效期和公私钥匹配。
4. 从空目录重新构建 v2grad 版本，确认没有 ssl 目录；上传到新的 releases/<版本时间>，不使用仓库原有 public 中可能残留的旧 SSL 文件，不使用 rsync --delete。为文件配置 Nginx worker 可读权限。
5. 将 current 链接指向该版本。后续更版用临时链接和 GNU mv -T 原子替换，不能将 cp 覆盖在用目录作为发布方式。
6. 安装 v2grad.conf 到 sites-available，新增 sites-enabled 链接；原 default 暂时保留。执行 nginx -t，通过后 reload，仅启用新虚拟主机。
7. 先不改公共 A 记录，使用 curl --resolve 对实际服务器的新站做 HTTPS 验收；这次必须使用可信生产证书，不能加 -k。核对主页、代表文章、图片、canonical、sitemap、404。
8. 验收通过后，配置新域名 @ 的 A 指向现有服务器、www 的 CNAME 指向新主域名；保留已有 Google TXT 和 Bing 专用 CNAME。实际生效等待取决于 DNS 缓存。
9. 用 imzhanghao.conf 替换 sites-available/default；保留现有 default 符号链接。不能额外再启用一份旧主站 server，避免 default_server 与 server_name 冲突。执行 nginx -t，通过后 reload。
10. 验收旧站四种入口、重要历史外链、新站 HTTPS、资源、robots、报告和新旧证书。两平台的 sitemap、Google 地址更改仍按迁移方案在技术验收后办理。

服务器上 `current` 的首次创建与后续切换示例（仅适用于 Linux 服务器；目录及版本已准备）：

```bash
# 首次发布：current 尚不存在时
ln -s /var/www/v2grad/releases/<版本时间> /var/www/v2grad/current

# 后续切换：使用未占用的临时链接名，保留已有版本目录
ln -s /var/www/v2grad/releases/<版本时间> /var/www/v2grad/current.next
mv -Tf /var/www/v2grad/current.next /var/www/v2grad/current
nginx -t
systemctl reload nginx
```

生产连通性验证示例（在本机执行；本文没有执行）：

```bash
curl --resolve v2grad.com:443:47.105.106.15 -I https://v2grad.com/
curl --resolve v2grad.com:443:47.105.106.15 -I https://v2grad.com/2022/06/10/mlops/
curl --resolve www.v2grad.com:443:47.105.106.15 -I https://www.v2grad.com/2022/06/10/mlops/
```

GitHub Pages 的远端自定义域名绑定本次未检查、未更改。此处仅使用自有 Nginx 发布链路，GitHub master 保留；以后如继续同步它，再单独确认绑定和发布范围。

## 回退步骤

### 新站自身故障，旧站尚未启用 301

保持旧站原样。若尚未配置新域名 A 记录，延后切换即可；已启用的新站若有上一份可用版本，切回 current 的上一目标。配置问题恢复本次备份后 nginx -t，通过才 reload。

### 已启用旧站 301 后出现大面积故障

1. 优先让新站恢复为可用版本，因为客户端可能已缓存旧站的 301。
2. 需要恢复旧主站时，将 `imzhanghao.rollback.conf` 安装为 sites-available/default，检查 /var/www/hexo 和独立旧证书仍可用，执行 nginx -t，通过后 reload。
3. 新站继续返回可用内容，不立即反向 301 回旧域名，避免“缓存的旧站→新站”和“新站→旧站”形成循环。应保留新域名 DNS、证书和服务。
4. 若已经向 Google 提交地址更改，按官方取消流程处理；重新以哪个域名作为主站、canonical 如何恢复以及是否最终反向迁移，另行确认后执行。不要因几天搜索排名波动触发回退。

回退候选配置保留证书隔离和 /ssl/ 404，不恢复原来从网站目录读取密钥的做法。原始配置也已在本地保留为只读证据；生产执行前还要现场备份，不能用历史文件覆盖别人后续做的变更。

## 本地验证结果与边界

使用官方 nginx:1.14-alpine 镜像，实际版本为 1.14.2；与线上 1.14.0 同属 1.14 系列，Ubuntu 的编译模块可能不同。迁移与回退两套配置均通过 nginx -t，正式启用前仍要在目标服务器执行 nginx -t。

隔离测试只绑定本机 18081、18444；18080 已被占用，未影响占用它的服务。TLS 使用本地生成的短期测试证书，没有下载或使用线上私钥。测试后两个临时容器均停止、自动移除。

| 测试 | 结果 |
| --- | --- |
| 127 个旧 URL × 主域名/www × HTTP/HTTPS | 508/508 返回 301，Location 为新站同一路径 |
| 新站全部映射路径 | 127/127 返回 200 |
| 其他迁移检查 | 31 项通过，覆盖新域名规范化、编码路径及参数、旧 robots、HTTP ACME、404、SSL 目录隔离和报告例外 |
| 回退旧站 | 127/127 路径恢复 200 |
| 回退代表检查 | www 回旧主站、新站保留 200、SSL 路径 404、原报告可用 |

证据位于 `/Users/zhanghao/.codex/artifacts/v2grad/20261111/nginx/`，其中包含原 default 和钩子快照、迁移与回退验证 JSON。未执行生产部署、reload、DNS 调整、真实新证书签发、生产证书握手或搜索回迁。

新增资源为一份约 13 MB 的静态站、少量配置与日志，无需新服务器。保留两三个版本按几十 MB 估算。现有 logrotate 覆盖 `/var/log/nginx/*.log`，每日轮转、保留 14 份并压缩，已覆盖新日志命名。

检查结束时再次比对服务器 default 和 post-receive 的 SHA-256，与本地读取快照相同；本次没有改写这两个文件。

## 参考

- [Nginx return 指令](https://nginx.org/en/docs/http/ngx_http_rewrite_module.html#return)
- [Nginx request_uri 变量](https://nginx.org/en/docs/http/ngx_http_core_module.html#var_request_uri)
- [Certbot 手动验证与续期说明](https://eff-certbot.readthedocs.io/en/stable/using.html#manual)
