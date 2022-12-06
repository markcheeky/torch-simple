FROM python:3.11.0-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

RUN apt-get update && apt-get install -y \
    sudo \
    apt-utils \
    git \
    wget \
    curl \
    jq


ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
 && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN python -m pip install -U pip setuptools wheel poetry
RUN python -m poetry config virtualenvs.in-project true \
 && python -m poetry config virtualenvs.prompt "venv"
