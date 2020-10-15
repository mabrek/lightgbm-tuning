FROM python:3.8.6-buster

COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
