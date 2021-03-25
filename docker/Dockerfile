FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
LABEL maintainer="Akira Terauchi <akira090602@gmail.com>"

SHELL ["/bin/bash", "-c"]

ENV ROOTHOME /root
ENV WORKSPACE /var/www

RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libopencv-dev
# RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
RUN pip install poetry

# set poetry's path
ENV PATH $ROOTHOME/.poetry/bin:$PATH

COPY pyproject.toml $WORKSPACE
COPY poetry.lock $WORKSPACE

RUN pip install --ignore-installed ptyprocess
RUN pip install --ignore-installed pyyaml
RUN pip install --upgrade keyrings.alt
RUN poetry config virtualenvs.create false

RUN pip install --upgrade pip && \
    pip install -U setuptools && \
    poetry install -n

ENV USERNAME python
RUN groupadd -r $USERNAME
RUN useradd -r -g $USERNAME $USERNAME
# RUN chown $USERNAME:$USERNAME -R $ROOTHOME

ENV DEBCONF_NOWARNINGS yes
RUN apt-get update && apt-get install -y locales
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.UTF-8  
ENV LC_ALL ja_JP.UTF-8

# USER $USERNAME
WORKDIR workspace

CMD ["python"]
