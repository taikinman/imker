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
You don't need to learn a lot of package-specific terms and manner to define your pipeline. Each task used for imker are defined with sklearn-like interface. Existing sklearn modules are also used in imker as it is. Then, each task is connected to other tasks with PyTorch-like interface. The pipeline consists 5 components, PreProcessor, Splitter, OOFPreProcessor, Model, and PostProcessor. You can intuitively create a complex pipeline.
</details>

<details>
<summary>High modularity of task</summary>
A sklearn-like task provides reusability in other places or other projects. Here is an example of a user-defined task. Even if the dependencies between other tasks is changed, in many cases, you don't need to modify the source code of the task because of modularity.

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
PyTorch-like interface allows user to understand how pipeline perform without large effort even if pipeline grows complex. Here is an example of the preprocessing components of the pipeline with the titanic dataset.

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
Imker can easily control how to cache outputs from each task. If you want to obtain specific intermediate outputs in the desired format, you can do so. When you don't need to cache outputs due to process speed or storage capacity considerations, you can choose whether or not to cache task-by-task basis. These behaviors can be specified through the TaskConfig.

To cache the results of transform(), predict() or predict_proba(), you just pass True to argument cache.

```python
Task(TaskConfig(task=..., 
                cache=True
                ))
```

By default, a hash is generated from the source code, input data, and the parameters of the task. A cached file is a compressed file of a pickled object, and the default format is pbz2. If you want to change the format, you can pass another processor as an argument to the cache_processor of the TaskConfig. You can also specify your custom cache processor.

</details>

<details>
<summary>Ease of creating inference workflow</summary>
The sklearn interface can easily create the inference workflow, so imker can also do it. Once you fit your pipeline to your dataset, you just run the inference method for test data as shown below. You don't need to separate the training workflow and the inference workflow.

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
