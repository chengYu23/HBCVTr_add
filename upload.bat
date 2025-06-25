@echo off
REM 设置用户信息（只需执行一次，已经设置过可注释掉）
@REM git config --global user.name "你的名字"
@REM git config --global user.email "你的邮箱@example.com"

REM 初始化 Git 仓库
git init

REM 添加远程仓库（请替换为你自己的 GitHub 地址）
git remote add origin https://github.com/chengYu23/HBCVTr_add.git

REM 创建 main 分支并切换
git push -u origin main --force

REM 添加 .gitignore 忽略 .pt 文件
echo *.pt > .gitignore

REM 添加所有文件
git add .

REM 首次提交
git commit -m "Initial commit"

REM 推送到 GitHub main 分支
git push -u origin main

pause
