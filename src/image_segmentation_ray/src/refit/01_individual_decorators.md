### Inject Object

Keyword: `object`

The inject object decorator will handle any incoming dictionaries and parse them for
`object` definitions, before passing it to the function. When dealing with pipelines
such as `kedro`, `parameters.yml` is where parameters may be defined. In YAML for example,
objects cannot be defined, but this way, one can define objects declaratively in YAML,
keeping the underlying code cleaner.

The key principle here is dependency injection, or inversion of control. According to
this principle, a class (or function) should concentrate on fulfilling its
responsibilities and not on creating objects that it requires to fulfill those
responsibilities (https://www.freecodecamp.org/news/a-quick-intro-to-dependency-injection-what-it-is-and-when-to-use-it-7578c84fa88f/).

Let's demonstrate the impact this has on the code and how you can call a function:
```
def bad_fit_data(
    data: pd.DataFrame, X: List[str], y: str, model_str: str
) -> BaseEstimator:
    """Fit a model on data.

    Args:
        data: A pandas dataframe.
        X: A list of features.
        y: The column name of the target.
        model_str: The string path to the model object.

    Returns:
        A fitted model object.
    """
    model_object = load_obj(model_str)
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object

# calling the "bad" example:
bad_fitted_model_object = bad_fit_data(
    data=df,
    X=["feature_1"],
    y=["target"],
    model="sklearn.linear_model.LinearRegression",
)
```


```python
# let's define a cleaner example
from typing import List

import pandas as pd
from sklearn.base import BaseEstimator

from refit.v1.core.inject import inject_object


@inject_object()
def fit_data(
    data: pd.DataFrame, X: List[str], y: str, model_object: BaseEstimator
) -> BaseEstimator:
    """Fit a model on data.

    Args:
        data: A pandas dataframe.
        X: A list of features.
        y: The column name of the target.
        model_object: The sklearn model type to use.

    Returns:
        A fitted model object.
    """
    fitted_model_object = model_object.fit(data[X], data[y])
    return fitted_model_object

df = pd.DataFrame(
        [
            {"feature_1": 1, "target": 1},
            {"feature_1": 2, "target": 2},
            {"feature_1": 3, "target": 3},
        ]
    )

from sklearn.linear_model import LinearRegression

# usage as normal in code
fitted_model_object = fit_data(
    data=df, X=["feature_1"], y=["target"], model_object=LinearRegression()
)

# parametrised via dictionary to be used in pipeline
fitted_model_object = fit_data(
    data=df,
    X=["feature_1"],
    y=["target"],
    model_object={"object": "sklearn.linear_model.LinearRegression"},
)
```
#### Example
The example function:


```python
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer

from refit.v1.core.inject import inject_object


@inject_object()
def fit_then_predict(data: pd.DataFrame, imputer: _BaseImputer) -> pd.DataFrame:
    imputed_data = imputer.fit_transform(data)

    return pd.DataFrame(imputed_data, columns=data.columns)
```
The dataset:


```python
data = pd.DataFrame([{"c1": 1}, {"c1": None}])

print(data)
```
        c1
    0  1.0
    1  NaN

##### Example 1 - Running Simple Imputer
Running using code:


```python
imputed_data = fit_then_predict(data=data, imputer=SimpleImputer())
print(imputed_data)
```
        c1
    0  1.0
    1  1.0

The decorator has no effect.


Running using parameters:


```python
imputed_data = fit_then_predict(
    data=data, imputer={"object": "sklearn.impute.SimpleImputer"}
)
print(imputed_data)
```
        c1
    0  1.0
    1  1.0

##### Example 2 - Running KNNImputer
Running using code:


```python
imputed_data = fit_then_predict(data=data, imputer=KNNImputer(n_neighbors=2))
print(imputed_data)
```
        c1
    0  1.0
    1  1.0

Running using parameters:


```python
imputed_data = fit_then_predict(
    data=data, imputer={"object": "sklearn.impute.KNNImputer", "n_neighbors": 2}
)
print(imputed_data)
```
        c1
    0  1.0
    1  1.0

Notice that kwargs can be declared within the dictionary.



#### Further Usage Examples

The decorator will scan any dictionary passed to a function for the
keywords: `{"object": "path.to.definition"}`.

For classes, init arguments may be passed like so:

    {
        "object": "path.to.MyClass",
        "class_arg1": "value",
        "class_arg2": "another_value"
    }

This is equivalent to
`MyClass(class_arg1="value", "class_arg2="another_value")` assuming:

    class MyClass:
        def __init__(class_arg1, class_arg2):
           ...

The keyword `instantiate` may be used to prevent the decorator from
instantiating the class:

    {
        "object": "path.to.MyClass",
        "instantiate": False
    }

This is equivalent to `MyClass` (or `x = MyClass`) without the
parantheses which you can instantiate later in your own code. If
`instantiate: False` is supplied, class init arguments have no effect.
If the `instantiate: False` argument is supplied, it is recommended to
remove init arguments.

To parametrize for functions, simply provide the path to the function:

    {
        "object": "path.to.my_function"
    }

This is equivalent to `x = my_function`.

Passing an argument to a function defined as such will cause the
function to be evaluated:

    def my_function(x):
        return x

    {
        "object": "path.to.my_function",
        "x": 1
    }

This is equivalent to `result = my_function(x=1)`. Additional arguments
should be passed via use of keywords, corresponding to actual function
signature.

Another example:

    {
        "object": "pyspark.sql.functions.max",
        "col": "column_1"
    }

This is equivalent to `f.max("column_1")`.

#### Advanced Usage Examples

The decorator also handles nested structures (via recursion):

    # nested class example where objects defined need other objects
    {
        "object": "path.to.MyClass":
        "some_init_arg_that_requires_another_object": {
            "object": "path.to.that.Object",
            "another_init_that_requires_yet_another_object": {
                "object": "path.to.yet.another.Object"
            }
        }
    }

    # nested function example where function requires an object.\
    {
        "object": "path.to.my_func":
        "my_func_args": {
            "object": "path.to.a.Class"
        }
    }

The recommended rule of thumb is to look at the original code, make sure
that is clean and easy to read. In most cases, if the input parameter is
heavily nested, this might indicate the original code could benefit from
refactoring to flatten out logic.

The decorator allows us to exclude certain keywords if we want to delay their injection.
To do this you would use the `exclude_kwargs` parameter. This expects a list of keywords to exclude.

To see this in action, consider a situation where you are calling a function that needs to serialize some
parameters that also use `refit` syntax.

    @inject_object(exclude=['params_to_serialize'])
    def serialize_object(writer: AbstractWriter, params_to_serialize: Dict[str, str]):
      writer.write(path, params_to_serialize)

We can call this function using the following config:

    {
        "writer": {
            "object": "path.to.some.abstract.writer"
            "save_path": "some_path"
            "compress": True
        },
        "params_to_serialize": {
            "object": "path.to.MyClass":
            "some_init_arg_that_requires_another_object": {
                "object": "path.to.that.Object",
                "another_init_that_requires_yet_another_object": {
                    "object": "path.to.yet.another.Object"
                }
            }
        }
    }

In this instance, only the `AbstractWriter` class will be injected, whilst inside the function,

    params_to_serialize :Dict = {
        "object": "path.to.MyClass":
        "some_init_arg_that_requires_another_object": {
            "object": "path.to.that.Object",
            "another_init_that_requires_yet_another_object": {
                "object": "path.to.yet.another.Object"
            }
        }
    }

### Defender `has_schema`

The ever popular `defender` style `has_schema`:



Assuming we have the following dataframe:


```python
spark_df.show(truncate = False)
```
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+
    |int_col|long_col|string_col    |float_col|double_col|date_col  |datetime_col             |array_int|
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+
    |1      |2       |awesome string|10.01    |0.89      |2012-05-01|2022-03-09 17:15:32.56365|[1, 2, 3]|
    |2      |2       |null          |10.01    |0.89      |2012-05-01|2022-03-09 17:15:32.56365|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+
    

And assuming we have the following function:


```python
import pyspark.sql.functions as f

@has_schema(
    schema={
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
        "new_output": "int",
    },
    allow_subset=True,
    raise_exc=True,
    relax=False,
)
def my_func(df):
    df_new = df.withColumn("new_output", f.lit(1))
    return df_new
```
We can run our function like so and the schema checking will be performed:


```python
result = my_func(spark_df)
result.show(truncate = False)
```
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+----------+
    |int_col|long_col|string_col    |float_col|double_col|date_col  |datetime_col             |array_int|new_output|
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+----------+
    |1      |2       |awesome string|10.01    |0.89      |2012-05-01|2022-03-09 17:15:32.56365|[1, 2, 3]|1         |
    |2      |2       |null          |10.01    |0.89      |2012-05-01|2022-03-09 17:15:32.56365|[1, 2, 3]|1         |
    +-------+--------+--------------+---------+----------+----------+-------------------------+---------+----------+
    

The decorator works with both pandas and spark dataframes.



The pandas dataframe:


```python
print(pd_df)
```
       float_col  int_col datetime_col    date_col string_col datetime_ms_col
    0        1.0        1   2018-03-10  2018-03-10        foo      2018-03-10
    1        2.0        2   2018-04-10  2018-04-10        bar      2018-04-10

Assuming we have the following function:


```python
@has_schema(
    schema={
        "int_col": "int64",
        "string_col": "object",
        "float_col": "float64",
        "date_col": "object",
        "datetime_col": "datetime64[ns]",
        "new_output": "int64",
        "datetime_ms_col": "datetime64[ns]",
    },
    allow_subset=False,
    raise_exc=True,
)
def pandas_example(df):
    df["new_output"] = 1
    return df
```
Running gives us:


```python
result = pandas_example(pd_df)
print(result)
```
       float_col  int_col datetime_col    date_col string_col datetime_ms_col  \
    0        1.0        1   2018-03-10  2018-03-10        foo      2018-03-10   
    1        2.0        2   2018-04-10  2018-04-10        bar      2018-04-10   
    
       new_output  
    0           1  
    1           1  

### Has Schema

Check the input schema of a dataframe, but with keyword injection of instead of the
`Defender` style.


```python
from refit.v1.core.has_schema import has_schema


@has_schema()
def node_func(df):
    return df

_input_has_schema = {
    "df": "df",
    "expected_schema": {
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    "allow_subset": False,
    "raise_exc": True,
    "relax": False,
}
_output_has_schema = {
    "expected_schema": {
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    "allow_subset": False,
    "raise_exc": True,
    "relax": False,
    "output":0,
}

node_func(
    df=spark_df,
    input_has_schema=_input_has_schema,
    output_has_schema=_output_has_schema
)

```

    DataFrame[int_col: int, long_col: bigint, string_col: string, float_col: float, double_col: double, date_col: date, datetime_col: timestamp, array_int: array<int>]


For multiple inputs or outputs, pass in a list of dictionaries instead:


```python
@has_schema()
def node_func(df1, df2):
    return df1, df2

_input_has_schema = [
    {
        "df": "df1",
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
    },
    {
        "df": "df2",
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
    }
]
_output_has_schema = [
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": 0,
    },
    {
        "expected_schema": {
            "int_col": "int",
            "long_col": "bigint",
            "string_col": "string",
            "float_col": "float",
            "double_col": "double",
            "date_col": "date",
            "datetime_col": "timestamp",
            "array_int": "array<int>",
        },
        "allow_subset": False,
        "raise_exc": True,
        "relax": False,
        "output": 1,
    }
]
node_func(
    df1=spark_df,
    df2=spark_df,
    input_has_schema=_input_has_schema,
    output_has_schema=_output_has_schema
)
```

    (DataFrame[int_col: int, long_col: bigint, string_col: string, float_col: float, double_col: double, date_col: date, datetime_col: timestamp, array_int: array<int>],
     DataFrame[int_col: int, long_col: bigint, string_col: string, float_col: float, double_col: double, date_col: date, datetime_col: timestamp, array_int: array<int>])


### Retry

The retry decorator retries a function given a list of exceptions. This may be useful
fo functions that are calling APIs.


```python
from refit.v1.core.retry import retry

@retry()
def add_retry_column(*args, **kwargs):
    def add_column(df):
        df["retry_col"] = 1
        return df
    return add_column(*args, **kwargs)

@retry()
def dummy_func_raises(*args, **kwargs):
    def dummy_raises(x):
        print("x is", x)
        raise TypeError("random TypeError")
    return dummy_raises(*args, **kwargs)
```
Running the following should show the behavior of the decorator, when the expected error
is raised you see the statement printed for each retry.


```python
dummy_func_raises(
    retry={"exception": [TypeError], "max_tries": 5, "interval": 0.01},
    x=2
)
```
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.

    x is 2
    x is 2
    x is 2
    x is 2
    x is 2

Since no error is raised in the following snippet, the decorator doesn't retry.


```python
add_retry_column(
    retry={"exception": [TypeError], "max_tries": 5, "interval": 0.01},
    df=pd_df
)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
      <td>2018-03-10</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Input kwarg filter

This decorator modifies the function definition to include an additional filter based on input kwargs.
This works for both spark and pandas dataframes.

A node that does purely a filter and does IO is a low value node. There is cost from
maintaining extra entries in the catalog and pipeline. However, converting the dataset
(if kedro) to a `MemoryDataSet` breaks integration, as nodes should be re-runnable on
their own.

An alternative is to modify the source code to add a filter, but this also dilutes
the original logic, making it harder to read.


```python
from refit.v1.core.input_kwarg_filter import add_input_kwarg_filter

@add_input_kwarg_filter()
def my_node_func(df):
    return df

spark_df.count()

result1 = my_node_func(df=spark_df, kwarg_filter={"df":"int_col != 1"})

result1.count()

pd_df.head()

result2 = my_node_func(df=pd_df, kwarg_filter={"df":"string_col != 'foo'"})

result2.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Input kwarg select

This decorator modifies the function definition to include an additional select based on input kwargs.
This works only for dataframes.

A node that does purely a select and does IO is a low value node. There is cost from
maintaining extra entries in the catalog and pipeline. However, converting the dataset
(if kedro) to a `MemoryDataSet` breaks integration, as nodes should be re-runnable on
their own.

An alternative is to modify the source code to add a select, but this also dilutes
the original logic, making it harder to read.


```python
from refit.v1.core.input_kwarg_select import add_input_kwarg_select

@add_input_kwarg_select()
def my_node_func(df):
    return df

spark_df.show()

result1 = my_node_func(df=spark_df, kwarg_select={"df":["length(string_col)"]})

result1.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +------------------+
    |length(string_col)|
    +------------------+
    |                14|
    |              null|
    +------------------+
    

### Unpack params
This decorator unpacks the top level dictionaries in args by 1 level.
Most beneficial if used as part of a kedro node.


```python
from refit.v1.core.unpack import unpack_params

@unpack_params()
def my_func_pd(*args, **kwargs):
    def dummy_func(df, x, y):
        df["unpack"] = df["int_col"] + x + y
        return df

    return dummy_func(*args, **kwargs)


result = my_func_pd(unpack={"df": pd_df, "x": 1, "y": 0})

print(result)
```
       float_col  int_col datetime_col    date_col string_col datetime_ms_col  \
    0        1.0        1   2018-03-10  2018-03-10        foo      2018-03-10   
    1        2.0        2   2018-04-10  2018-04-10        bar      2018-04-10   
    
       new_output  retry_col  unpack  
    0           1          1       2  
    1           1          1       3  

The decorator unpacks the params and we get the expected result.


One can remove the unpack keyword when not using as part of the
node decorator.


```python
try:
   result = my_func_pd({"df": pd_df, "x": 1, "y": 0})

except TypeError as e:
    print(e)

```
    dummy_func() missing 2 required positional arguments: 'x' and 'y'

When using the above with kedro nodes, it would be as follows:
Unpacks dictionary with key is `unpack`. Typically used to unpack a dictionary from
kedro parameters.

Example usage:


```python
# Unpack using args
@unpack_params()
def dummy_func(df, x, y, z):
    df["test"] = x + y - z
    return df

x = 1
param = {"unpack": {"y": 1, "z": 2}}
result = dummy_func(df, x, param)

# Unpack using kwargs
params = {"x": 1, "y": 1, "z": 2}
result = dummy_func(df=df, unpack=params)

# Unpack using args and kwargs
@unpack_params()
def dummy_func3(df, param1, param2, x, y, z,):
    df["col2"] = param1["col2"]
    df["col3"] =  param2["col3"]
    df["col4"] = x+y-z
    return df
params1 = {"param1":{"col2": 1}, "unpack": {"y": 2}}
params2 = {"param2":{"col3": 2}, "unpack": {"z": 3}}
result = dummy_func3(df, params1, params2, unpack={"x": 1})
```
### Output filter

This decorator modifies the function definition to include an additional filter to be applied on output dataframe. This works for both spark and pandas dataframes.


```python
from refit.v1.core.output_filter import add_output_filter

@add_output_filter()
def my_node_func(df):
    return df

spark_df.count()

result1 = my_node_func(df=spark_df, output_filter="int_col != 1")

result1.count()

pd_df.head()

result2 = my_node_func(df=pd_df, output_filter="string_col != 'foo'")

result2.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
      <th>unpack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


### Output primary key check

This decorator performs the primary key functionality (duplicate and not null checks) on
the set of columns we pass from an output dataframe (spark and pandas).

Below example shows the primary key check without allowing any null and
duplicate values in it. This can be achieved by setting `nullable = False`.
By default `nullable` option is False.


```python
from refit.v1.core.output_primary_key import add_output_primary_key

@add_output_primary_key()
def my_node_func(df):
    return df

spark_df.show()

try:
   result1 = my_node_func(df=spark_df, output_primary_key={"columns": ["int_col", "string_col"]})

except TypeError as error:
    print(error)

pd_df.head()

result2 = my_node_func(df=pd_df, output_primary_key={"columns": ["int_col"]})

result2.head()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    Primary key columns ['int_col', 'string_col'] has either duplicate values or null values.


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
      <th>unpack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
      <td>2018-03-10</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


Now check by setting option `nullable=True`, this allows null in composite key and without duplicate values in it.
However for single column of primary key, it won't allow duplicates and null values in it.


```python
@add_output_primary_key()
def my_node_func(df):
    return df

spark_df.show()

result1 = my_node_func(df=spark_df, output_primary_key={"nullable":True, "columns": ["int_col", "string_col"]})

result1.show()

pd_df.head()

result2 = my_node_func(df=pd_df, output_primary_key={"nullable":True, "columns": ["int_col", "string_col"]})

result2.head()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
      <th>unpack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
      <td>2018-03-10</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


### Defender primary key check

This decorator is the `defender` style primary key check  where a primary key
is defined as no duplicates and not nullable (by default). The decorator works for both
spark and pandas dataframes.

By default, the decorator will check the output dataframe's primary key:


```python
from refit.v1.core.defender_primary_key import primary_key

@primary_key(
    primary_key=["int_col"],
)
def node_func(df):
    return df

result_df = node_func(df=spark_df)

spark_df.show()
result_df.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

However, to check input dataframes, leverage the `df` argument:


```python
@primary_key(
    df="my_df",
    primary_key=["int_col"],
)
def node_func(my_df):
    return my_df

result_df = node_func(my_df=spark_df)

spark_df.show()
result_df.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

#### Nullable Primary Key

Sometimes, when it comes to composite primary keys, we may want to allow nullables in
some of the columns (with the exception of all keys being null). We can do so using the
`nullable` argument:


```python
@primary_key(
    primary_key=["int_col", "string_col"],
    nullable=True
)
def node_func(df):
    return df

result_df = node_func(df=spark_df)

spark_df.show()
result_df.show()

```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

Note that by default `nullable` is False as we don't expect nulls in any part of a
primary key (composite or not). This function is particularly useful when it comes to
data exploration during the early stages.


#### Function With Multiple Returned Variables
If a function returns multiple variables, you may leverage the `output` kwarg like so:


```python
@primary_key(
    primary_key=["int_col", "string_col"],
    output=1, # or 0
    nullable=True
)
def node_func(df):
    return [df, df]

result_df1, result_df2 = node_func(df=spark_df)

spark_df.show()
result_df1.show()
result_df2.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|          null|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

### Make list regexable

Keyword: `No Keyword`

The usage is to decorate your core function (as opposed to being used with `@augment`).
This decorator allows you to provide a list of regex in your function when selecting columns from a dataframe
(pandas and spark).
We have implemented it defender style because this is more modifying the core function.
It's not something we want to be able to configure on the fly at node level.
It's conforming to the dependency inversion principle where you expect your required dependencies
(in this case a list of columns after regex from the dataframe).

The `@make_list_regexable` decorator gives you the ability to pass regex in your list when it comes to selecting columns from a dataframe.
The decorator requires two parameters, a source dataframe which has actual columns and a list which might contains
regex.
If the list is empty or no list is provided the decorator will not do any modifications, normal course will follow.


```python
from refit.v1.core.make_list_regexable import make_list_regexable


@make_list_regexable(source_df = "df", make_regexable="param_keep_cols")
def accept_regexable_list(df, param_keep_cols, enable_regex):
    df = df[[*param_keep_cols]]
    return df

data = pd.DataFrame(
        [
            {"feature_1": 1, "target": 1},
            {"feature_1": 2, "target": 2},
            {"feature_1": 3, "target": 3},
        ]
    )

final_result = accept_regexable_list(
    df=data,
    param_keep_cols=["feature_.*"],
    enable_regex=True,
)

print(final_result)

```
       feature_1
    0          1
    1          2
    2          3

### Fill nulls

This decorator modifies the function definition to fill the nulls with a
given value on the output dataframe. This works for both spark and
pandas dataframes.

Here is an example with a spark dataframe:


```python
from refit.v1.core.fill_nulls import fill_nulls
@fill_nulls()
def my_node_func(df):
    return df
result1 = my_node_func(df=spark_df, fill_nulls={"value": "default_value", "column_list": ["string_col"]})
result1.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2| default_value|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

Here is an example with a pandas dataframe:


```python
pd_df2 = pd.DataFrame(
    {
        "float_col": [1.0, 2.0, None],
        "int_col": [1, None, 2],
        "datetime_col": [pd.Timestamp("20180310"), pd.Timestamp("20180410"), pd.Timestamp("20180510")],
        "date_col": [pd.Timestamp("20180310").date(), pd.Timestamp("20180410").date(), pd.Timestamp("20180510").date()],
        "string_col": ["foo", "bar", None],
    }
)
pd_df2.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>2018-05-10</td>
      <td>2018-05-10</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


The column list can be a regex string, when `enable_regex` is set to True. eg: The following example takes all the columns.
In this case, pandas will fill in other columns because pandas allows mix types. This is likely
to cause other issues down the line, for example writing to parquet. Be careful when using the
regex functionality!


```python
result2 = my_node_func(df=pd_df2, fill_nulls={"value": "default_value", "enable_regex": True, "column_list": [".*"]})
result2.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>default_value</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>default_value</td>
      <td>2.0</td>
      <td>2018-05-10</td>
      <td>2018-05-10</td>
      <td>default_value</td>
    </tr>
  </tbody>
</table>
</div>


Here is another pandas example, which only fills the specified columns:


```python
pd_df2 = pd.DataFrame(
    {
        "float_col": [1.0, 2.0, None],
        "int_col": [1, None, 2],
        "datetime_col": [pd.Timestamp("20180310"), pd.Timestamp("20180410"), pd.Timestamp("20180510")],
        "date_col": [pd.Timestamp("20180310").date(), pd.Timestamp("20180410").date(), pd.Timestamp("20180510").date()],
        "string_col": ["foo", "bar", None],
    }
)
result3 = my_node_func(df=pd_df2, fill_nulls={"value": 200, "column_list": ["int_col"]})
result3.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2018-03-10</td>
      <td>2018-03-10</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>200.0</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>2018-05-10</td>
      <td>2018-05-10</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>


### Remove Debug Columns
This may be leveraged with parquet's columnar format, where the previous function
stores columns with `_` prefix to disk, but the next node only selects and passes
through columns without this `_` prefix. Using this pattern, we can choose to persist
more columns and be able to investigate if anything looks wrong without having to re-run
the node, yet not suffer from performance implications downstream.
Note that this efficiently depends on how the underlying dataframe library reads in parquet files.


```python
from refit.v1.core.remove_debug_columns import (
    remove_input_debug_columns,
    remove_output_debug_columns,
)
PREFIX = "_"

df = pd.DataFrame(
    columns=['col1','_col2','col3'],
    data = [
        (1,2,3)
    ]
)

@remove_input_debug_columns()
@remove_output_debug_columns()
def some_func1(df):
    df['col4']=4
    df['_col5']=5
    return df
res = some_func1(df)
res.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col3</th>
      <th>col4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


### Augment

The `augment` decorator is a convenience decorator for multiple functionality listed
above. So you can simply do:


```python
from refit.v1.core.augment import augment

@augment()
def my_function(df, params):
    # do something
    return df
```
Instead of:


```python
@inject_object()
@retry()
@has_schema()
@add_input_kwarg_filter()
@add_output_filter()
@unpack_params()
def my_function(df, params):
    # do something
    return df
```
Note that not all individual decorators are encompassed within the `@augment` decorator.

Calling individual decorators using `augment`:

#### Inject:


```python
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute._base import _BaseImputer

from refit.v1.core.augment import augment


@augment()
def fit_then_predict(data: pd.DataFrame, imputer: _BaseImputer) -> pd.DataFrame:
    imputed_data = imputer.fit_transform(data)

    return pd.DataFrame(imputed_data, columns=data.columns)
```
The dataset:


```python
data = pd.DataFrame([{"c1": 1}, {"c1": None}])

print(data)
```
        c1
    0  1.0
    1  NaN

Imputed dataset:


```python
imputed_data = fit_then_predict(
    data=data, imputer={"object": "sklearn.impute.SimpleImputer"}
)
print(imputed_data)
```
        c1
    0  1.0
    1  1.0

#### Retry:


```python
@augment()
def dummy_func_raises(*args, **kwargs):
    def dummy_raises(x):
        print("x is", x)
        raise TypeError("random TypeError")
    return dummy_raises(*args, **kwargs)


dummy_func_raises(
    retry={"exception": [TypeError], "max_tries": 5, "interval": 0.01},
    x=2
)
```
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.
    ERROR:root:Hit exception: random TypeError. Sleeping: 0.01 seconds.

    x is 2
    x is 2
    x is 2
    x is 2
    x is 2

#### Has schema:


```python
@augment()
def node_func(df):
    return df

input_has_schema = {
    "df": "df",
    "expected_schema": {
        "int_col": "int",
        "long_col": "bigint",
        "string_col": "string",
        "float_col": "float",
        "double_col": "double",
        "date_col": "date",
        "datetime_col": "timestamp",
        "array_int": "array<int>",
    },
    "allow_subset": False,
    "raise_exc": True,
    "relax": False,
}

node_func(df=spark_df, input_has_schema=input_has_schema)

```

    DataFrame[int_col: int, long_col: bigint, string_col: string, float_col: float, double_col: double, date_col: date, datetime_col: timestamp, array_int: array<int>]


#### Input kwarg filter:


```python
@augment()
def my_node_func(df):
    return df

spark_df.count()

result1 = my_node_func(df=spark_df, kwarg_filter={"df":"int_col != 1"})

result1.count()

pd_df.head()

result2 = my_node_func(df=pd_df, kwarg_filter={"df":"string_col != 'foo'"})

result2.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
      <th>unpack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


#### Output filter


```python
result3 = my_node_func(df=spark_df, output_filter="int_col != 1")

result3.count()

pd_df.head()

result4 = my_node_func(df=pd_df, output_filter="string_col != 'foo'")

result4.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>float_col</th>
      <th>int_col</th>
      <th>datetime_col</th>
      <th>date_col</th>
      <th>string_col</th>
      <th>datetime_ms_col</th>
      <th>new_output</th>
      <th>retry_col</th>
      <th>unpack</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>2</td>
      <td>2018-04-10</td>
      <td>2018-04-10</td>
      <td>bar</td>
      <td>2018-04-10</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


#### Unpack params


```python
@augment()
def my_func_pd(*args, **kwargs):
    def dummy_func(df, x, y):
        df["unpack"] = df["int_col"] + x + y
        return df

    return dummy_func(*args, **kwargs)


result = my_func_pd(unpack={"df": pd_df, "x": 1, "y": 0,})

print(result)
```
       float_col  int_col datetime_col    date_col string_col datetime_ms_col  \
    0        1.0        1   2018-03-10  2018-03-10        foo      2018-03-10   
    1        2.0        2   2018-04-10  2018-04-10        bar      2018-04-10   
    
       new_output  retry_col  unpack  
    0           1          1       2  
    1           1          1       3  

#### Fill nulls params


```python
result5 = my_node_func(df=spark_df, fill_nulls={"value": "default", "column_list": ["string_col"]})

result5.show()
```
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |int_col|long_col|    string_col|float_col|double_col|  date_col|        datetime_col|array_int|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    |      1|       2|awesome string|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    |      2|       2|       default|    10.01|      0.89|2012-05-01|2022-03-09 17:15:...|[1, 2, 3]|
    +-------+--------+--------------+---------+----------+----------+--------------------+---------+
    

