FROM ubuntu:latest

MAINTAINER xniu@wpi.edu


RUN apt-get update \
&& apt-get install -y ssh \
&& apt-get install -y cron \
&& apt-get install -y expect \
&& apt-get install -y vim \
&& apt-get install -y software-properties-common \
&& apt-add-repository ppa:ansible/ansible \
&& apt-get install -y python3-pip python3-dev \
&& cd /usr/local/bin \
&& ln -s /usr/bin/python3 python \
&& pip3 install --upgrade pip \
&& apt-get update \
&& apt-get install -y ansible \
&& mkdir /Portfolio-RL \

WORKDIR /Portfolio-RL

ADD . /data
WORKDIR /code
ENTRYPOINT ["/bin/bash", "-c", "/start_only.sh"]