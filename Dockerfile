FROM python:3-alpine3.9
WORKDIR ./BackEnd
COPY . /BackEnd
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./BackEnd/app.py