# what is mlflow??
BI tools to support the ML lifecycle

## Role
- MLFlow tracking 
  - code shareing
- MLFlow Project
  - control environment and package
- MLFlow Models
  - deploy model

## install mlflow
```shell
pip install mlflow
```

## get Started
### Start mlflowUI
```shell
mlflow ui
```

### start ml
```shell
python ml_pipeline_mlflow.py
```


### deloy models
#### conda environment required
```shell
mlflow models serve --models-uri <runs:/25a4e6a7b9464661ae1a6be7f21a0e83/model> --port 1234
```
