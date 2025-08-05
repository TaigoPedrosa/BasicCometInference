FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["serve", "run", "config.yaml"]