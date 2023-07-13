# Imker
An easy-to-use ML pipeline package for the purpose of speeding up your experiments, which is inspired by scikit-learn and PyTorch interface. 

# Why Imker
## comparison summary with other ML pipeline tools
||Kedro|gokart|sklearn|Imker |
|--|--|--|--|--|
|learning cost|△|△|◎|◎|
|task modularity|〇|△|◎|◎|
|interactivity|△|〇|◎|◎|
|pipeline readability|△|〇|△|〇|
|cache flexibility|△|〇|△|◎|
|ease of creating inference workflow|△|△|◎|◎|

△：not good　〇：good　◎：better

<details>
<summary>Low learning cost</summary>
You don't need to learn a lot of package-specific terms and manner to define your pipeline. Each task used for imker are defined with sklearn-like interface. Existing sklearn modules are also used in imker as it is. Then each task are connected to other task with PyTorch-like interface. Pipeline has 5 components, PreProcessor, Splitter, OOFPreProcessor, Model, and PostProcessor. You can create complex pipeline intuitively.
</details>

<details>
<summary>High modularity of task</summary>
sklearn-like task provides reusability in other place or other project easily. Here is an example of user-defined task.

```python
class DropCols(BaseTask):
    def __init__(self, cols:list) -> None:
        self.cols = cols
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(self.cols, axis=1)
        return X
```

</details>

<details>
<summary>High readability</summary>
PyTorch-like interface allows user to understand how pipeline perform without large effort even if pipeline grows complex. Here is an example of preprocessing components of pipeline with titanic dataset.

```python
class PreProcessor(BaseProcessor):
    def __init__(self):
        self.drop = Task(TaskConfig(task=DropCols, 
                                    init_params={"cols":["name", "cabin", "ticket", 
                                                         "body", "boat", "home.dest"]}, 
                                    ))
        self.cat_encoder = Task(TaskConfig(task=OrdinalEncoder, # you can use scikit learn class as it is
                                           init_params={"handle_unknown":"use_encoded_value", 
                                                        "unknown_value":-1, 
                                                        "encoded_missing_value":-999}, 
                                            ))
        self.target_label_enc = Task(TaskConfig(task=LabelEncoder))
        self.dtype_converter = Task(TaskConfig(task=DTypeConverter, 
                                               init_params={"dtype":"int8"}))
        
    def forward(self, X, y=None):
        X = self.drop(X)
        X[["sex", "embarked"]] = self.cat_encoder(X[["sex", "embarked"]])
        X[["sex", "embarked"]] = self.dtype_converter(X[["sex", "embarked"]])
        if y is not None:
            y = self.target_label_enc(y) # target variable can be transformed as well as features
        return X, y
```

</details>

<details>
<summary>Highly flexible cache</summary>
Imker can control how-to-cache outputs from each task easily. If you want to get specific intermediate outputs with the format you want, you can do it. If you don't need to cache from the viewpoint of process speed or storage capacity, you can control whether cache or not, task by task. These behaviour can be specified through `TaskConfig` .
</details>

<details>
<summary>Ease of creating inference workflow</summary>
sklearn interface can easily create inference workflow. So imker also can do it. Once you fit your pipeline to your dataset, you just run inference method for test data as shown below. You don't need to separate training workflow and inference workflow.

```python
pipe.inference(X_test)
```

</details>



# Installation
## Requirements
- python >= 3.9

## Dependencies
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [pyyaml](https://github.com/yaml/pyyaml)

more details, see ./pyproject.toml

## using pip
```
pip install git+https://github.com/taikinman/imker.git
```

## using poetry
```
poetry add git+https://github.com/taikinman/imker.git
```

# Basic Usage
See ./example/*
