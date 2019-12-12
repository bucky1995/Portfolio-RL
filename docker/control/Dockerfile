FROM python:3.5

MAINTAINER xniu@wpi.edu

EXPOSE 5000

RUN apt-get update \
&& apt-get install -y ssh \
&& apt-get install -y cron \
&& apt-get install -y expect \
&& apt-get install -y vim \
&& apt-get install -y software-properties-common \
&& apt-add-repository ppa:ansible/ansible \
&& pip3 install --upgrade pip \
&& apt-get install -y ansible \
&& mkdir Portfolio-RL
#ENTRYPOINT ["/bin/bash", "-c", "/start_only.sh"]
CMD tail -f /dev/null