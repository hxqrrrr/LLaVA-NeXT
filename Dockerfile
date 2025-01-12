# 使用网易云PyTorch镜像
FROM hub.c.163.com/pytorch/pytorch:latest

WORKDIR /app

# 设置pip国内源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /root/.cache/huggingface

# 复制项目文件
COPY . /app/

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装 flash-attention
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# 安装项目
RUN pip install -e .

# 暴露端口
EXPOSE 7860

# 设置环境变量
ENV PYTHONPATH=/app
ENV HF_HOME=/root/.cache/huggingface

# 启动命令
CMD ["python", "llava/serve/gradio_web_server.py", "--host", "0.0.0.0", "--port", "7860"] 