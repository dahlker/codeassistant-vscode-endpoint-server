FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    curl \
    git \
    htop \
    tmux \
    vim \
    nano \
    python3.10 python3-pip \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" > /etc/profile.d/50-smc.sh
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN export LANG="en_US.UTF-8"
RUN pip install pipenv
RUN cd / && git clone https://github.com/dahlker/codeassistant-vscode-endpoint-server.git && cd /codeassistant-vscode-endpoint-server && pipenv install
RUN pip install huggingface-hub


CMD ["sh","-c", "huggingface-cli login --token=${token} && cd /codeassistant-vscode-endpoint-server && pipenv run python -m app.main"]

