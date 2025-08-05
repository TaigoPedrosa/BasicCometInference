FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends nvtop

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./config.yaml ./config.yaml

COPY . .

CMD ["serve", "run", "config.yaml"]