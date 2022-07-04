FROM python:3.8
COPY . ./
RUN pip3 install -r requirements.txt
CMD ["main_model_deployment.py"]
ENTRYPOINT ["python"]