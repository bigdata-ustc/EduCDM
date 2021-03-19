# CONTRIBUTE

[中文版本](CONTRIBUTE_CH.md)

## Guidance
Thank you for your interest in contributing to EduCDM! 
Before you begin writing code, it is important that you share your intention to contribute with the team, 
based on the type of contribution:

1. You want to propose a new feature and implement it.
    * Post about your intended feature in an issue, 
    and we shall discuss the design and implementation. 
    Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    * Search for your issue in the [EduCDM issue list](https://github.com/bigdata-ustc/CDM/issues).
    * Pick an issue and comment that you'd like to work on the feature or bug-fix.
    * If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, 
please submit a Pull Request to [EduCDM](https://github.com/bigdata-ustc/CDM).

The followings are some helpful guidelines for different types contribution:
 
### Add new dataset

If you want to add the data analysis or a new dataset, please submit a Pull Request to [EduData](https://github.com/bigdata-ustc/EduData).

### Add new CDM model

#### Dataset Processing

As for the dataset preprocessing, we suggest:

1. Write a script, and make sure that:
    - Processing and converting of the raw dataset.
    - Partitioning Training/validation/test dataset.
2. Provide or use [CDBD](https://github.com/bigdata-ustc/EduData) dataset (which is already divided into training/validation/test datasets).


#### Module

All modules are inherited from `Class CDM`, it will raise `NotImplementedError` if the functions are not implemented.

Note that we do not constrain your neural network or algorithms (for example, the network construction, optimizer, loss function definitions, etc.).

- **Train** module

This module is a training module, which is used to train model.

```python3
    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- **Eval** module

This module is a evaluation module, which is used to verify and test the model.

```python3
    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- **Save** module

This module is a model saving module, which is used to save the trained model.

```python3
    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

- **Load** module

This module is a model loading module, which is used to load the saved model.

```python3
    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError
```

#### Demo

Make sure you make a demo for your model. [An example]().

#### Docs Format

Numpy docs format is used:

```
function

    Parameters
    ----------
    Variable 1: type <int, float>, optional or not
       description
    Variable 2: type <int, float>, optional or not
       description
    ...

    Returns
    -------
    Variable: type <int, float>
       description

    See Also (Optional)
    --------
    Similar to function():

    Examples (Optional)
    --------
    >>> For example:
        ...
```

### About Commit

#### commit format

```
[<type>](<scope>) <subject>
```

#### type
- `feat`：新功能（feature）。
- `fix/to`：修复 bug，可以是 Q&A  发现的 bug，也可以是自己在使用时发现的 bug。
   - `fix`：产生 diff 并自动修复此问题。**适合于一次提交直接修复问题**。
   - `to`：只产生 diff 不自动修复此问题。**适合于多次提交**。最终修复问题提交时使用 `fix`。
- `docs`：文档（documentation）。
- `style`：格式（不影响代码运行的变动）。
- `refactor`：重构（即非新增功能，也不是修改 bug 的代码变动）。
- `perf`：优化相关，比如提升性能、体验。
- `test`：增加测试。
- `chore`：构建过程或辅助工具的变动。
- `revert`：回滚到上一个版本。
- `merge`：代码合并。
- `sync`：同步主线或分支的 bug。
- `arch`: 工程文件或工具的改动。

##### scope (optional)

scope 用于说明 commit 影响的范围，比如数据层、控制层、视图层等等，视项目不同而不同。

例如在Angular，可以是location，browser，compile，compile，rootScope， ngHref，ngClick，ngView 等。如果你的修改影响了不止一个 scope，你可以使用*代替。

##### subject (必须)

subject 是 commit 目的的简短描述，不超过50个字符。

结尾不加句号或其他标点符号。

#### Example

- **[docs] update the README.md**

## FAQ

Q: I have carefully test the code in my local system (all testing passed) but still failed in online CI?
 
A: There are two possible reasons: 
1. the online CI system is different from your local system;
2. there are some network error causing the downloading test failed, which you can find in the CI log.

For the second reason, all you need to do is to retry the test. 

