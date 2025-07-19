# Answer Book Project

📖 答案之书 📖

这是一个基于Gradio的交互式问答应用，基于《论语》和《道德经》选集构建的大模型 “答案之书”

## 环境要求
- Python 3.11
- pip (Python包管理器)
- Windows 10/11 操作系统
- Docker Desktop (用于运行Milvus向量数据库)

## 项目结构
- `use_gradio.py`: Gradio界面主程序
- `dataset_built.py`: 数据集构建工具
- `tool.py`: 通用工具函数
- `logger_config.py`: 日志配置
- `data/`: 存储项目数据
- `logs/`: 日志存储目录

## 安装与设置 (Windows系统)

### 1. 验证Python安装
```powershell
# 检查Python版本
python --version
# 应显示 Python 3.11.x

# 检查pip版本
pip --version
```

### 2. 创建并激活虚拟环境
```powershell
# 创建虚拟环境
conda create -n guoxue python=3.11

# 激活虚拟环境 
conda activate guoxue

# 激活成功后会看到命令行前出现 (guoxue) 标识
```

### 3. 安装依赖
```powershell
# 确保虚拟环境已激活
pip install -r requirements.txt
```

### 4. 安装Docker Desktop
#### 4.1 下载并安装Docker
1. 访问Docker官网下载页面: <https://www.docker.com/products/docker-desktop>
2. 双击安装文件，按照向导完成安装
3. 安装过程中确保勾选以下选项:
   - "Use WSL 2 instead of Hyper-V"
   - "Add shortcut to desktop"

#### 4.2 验证Docker安装
```powershell
# 启动Docker Desktop应用
# 打开PowerShell验证
docker --version
# 应显示Docker版本信息

docker-compose --version
# 应显示Docker Compose版本信息
```

### 5. 启动Milvus服务
#### 5.1 获取Milvus Docker Compose配置
```powershell
# 创建milvus目录并进入
mkdir -p d:\milvus && cd d:\milvus

# 下载配置文件
Invoke-WebRequest -Uri https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -OutFile docker-compose.yml
```

#### 5.2 启动Milvus服务
```powershell
# 启动服务 (后台运行)
docker-compose up -d

# 检查服务状态
docker-compose ps
# 应显示所有服务状态为 Up
```

#### 5.3 验证Milvus连接
```powershell
# 安装milvus-python客户端
pip install pymilvus==2.3.4

# 简单连接测试
python -c "from pymilvus import connections; connections.connect(host='localhost', port='19530'); print('Milvus connected successfully!')"
```

### 6. 配置应用连接Milvus
在项目根目录创建`config.yaml`文件，添加以下内容:
```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: ancient_chinese_literature
```

### 7. 运行应用
```powershell
# 跳转至answer_book文件夹
cd answer_book

# 加载数据到数据库
python dataset_built.py

# 然后运行gradio页面
python use_gradio.py
# 程序会自动启动浏览器并打开Gradio界面
# 如果没有自动打开，请访问终端中显示的本地地址（通常是 http://localhost:7860）
```

## 常见问题 (Windows)
- **Docker启动失败**: 确保已启用WSL2并安装Linux子系统
- **Milvus连接超时**: 检查Docker容器是否全部正常运行 (`docker-compose ps`)
- **端口占用**: Milvus默认使用19530端口，如冲突可修改docker-compose.yml中的端口映射
- **虚拟环境与Docker冲突**: 确保在激活虚拟环境前启动Docker服务
- **路径错误**: 在运行前修注意文件路径的正确
- **嵌入模型**: 若嵌入模型加载失败，可以从魔塔社区下载模型到本地调用

## 功能特点
- 交互式问答界面
- 古典文献内容展示
- 基于Milvus向量数据库的高效检索
- 实时处理与响应

## 许可证
[MIT](LICENSE)