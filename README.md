scibert
==============================

A short description of the project.



<p align="center">
<img src="docs/banner.webp"  />
</p>


Project Organization
------------
```
├── api
│   ├── app.py
│   ├── config.py
│   ├── resources
│   └── templates
├── Dockerfile
├── docs
│   ├── Analysis.md
│   ├── banner.webp
│   ├── DatasetDescription.md
│   ├── references.md
│   └── srs.md
├── LICENSE
├── logs
├── Makefile
├── notebooks
│   └── playground.ipynb
├── poetry.lock
├── poetry.toml
├── pyproject.toml
├── README.md
├── requirements.txt
├── run.sh
├── scibert
│   ├── config.py
│   ├── features
│   │   ├── build_features.py
│   ├── inference.py
│   ├── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── dispatcher.py
│   │   ├── __init__.py
│   │   ├── network.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── make_data.py
│   │   └── utils.py
│   ├── runs
│   └── utils
│       ├── decorators.py
│       ├── __init__.py
│       ├── logger.py
│       └── serializer.py
└── tests
    └── test_environment.py

```
--------


## Getting Started

### Requirements

```
pip install -r requirements.txt
```

### Download the dataset

The following command will download the dataset from the URL given in `src/config/config.py` file .

```
python -m scibert.data.make_dataset
```

### Run

```
python -m scibert.main
```
OR

```
./run.sh
```

### Test

```
python -m tests.test_environment
```


### To-do List

- [X] Download dataset
- [X] Pre-process data
- [X] Train model
- [X] Test model
- [X] Main Pipeline
- [X] Inference Pipeline


### Author

[Kamal Shrestha](https://shresthakamal.com.np/)

-------------------------------
