FROM continuumio/miniconda3:23.5.2-0-alpine AS builder

EXPOSE 8888

LABEL maintainer.name="mosdef-hub"\
  maintainer.url="https://mosdef.org"

ENV PATH /opt/conda/bin:$PATH

USER root

ADD . /gmso

WORKDIR /gmso

# Create a group and user
RUN addgroup -S anaconda && adduser -S anaconda -G anaconda

RUN apk update && apk add libarchive &&\
  conda update conda -yq && \
  conda config --set always_yes yes --set changeps1 no && \
  . /opt/conda/etc/profile.d/conda.sh && \
  conda install -c conda-forge mamba git && \
  mamba env create --file environment-dev.yml python=3.12 && \
  conda activate gmso-dev && \
  mamba install -c conda-forge jupyter && \
  pip install -e .&& \
  echo "source activate gmso-dev" >> \
  /home/anaconda/.profile && \
  conda clean -afy && \
  mkdir -p /home/anaconda/data && \
  chown -R anaconda:anaconda /gmso && \
  chown -R anaconda:anaconda /opt && \
  chown -R anaconda:anaconda /home/anaconda

WORKDIR /home/anaconda

COPY devtools/docker-entrypoint.sh /entrypoint.sh

RUN chmod a+x /entrypoint.sh

USER anaconda

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter"]
