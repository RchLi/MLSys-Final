# README

```unix
.
├── data
├── models
│   ├── svm.pkl
│   └── vec.pkl
├── pages
│   └── 1_Restaurant_Information.py
├── source
│   ├── final.ipynb
│   ├── final.py
│   ├── generate_data.ipynb
│   └── train.py
├── Hello.py
├── README.md
└── requirements.txt
```

## Installation

```shell
git clone git@github.com:RchLi/MLSys-Final.git
cd MLSys-Final
pip install -r requirements.txt
```

## Train Models

```shell
python source/train.py run
```

train.py is a metaflow file that pipelines the process of data processing and model training. It compares models' performance of different hyperparameters and stores the best one in models file.

## Generate Websites

```shel
streamlit run Hello.py
```

## Deploy on AWS

We deploy our project on AWS EC-V2, to access the website, use the [url](http://44.202.255.243:8501/)

The deployment follow the same procesure as is shown above. We first create a new instance of EC-V2, connect to with ssh and install our project. Note that we train our models locally and only transfer the best model to the server to reduce the storage cost.
