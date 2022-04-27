## Release History

#### 0.3.6
- Remove all upper bounds in requirements. 

#### 0.3.5
- Remove all .Rmd files.

#### 0.3.4
- Updated internals of primary key check decorators.

#### 0.3.3
- Convert .Rmd to .myst.

#### 0.3.2
- Add `remove_debug_columns` decorator.

#### 0.3.1
- Updated requirements and test requirements.

#### 0.3.0
- Updated `unpack_params` from boolean to kwarg to unpack.

#### 0.2.1
- Introduced docs folder to place Slack Blast poster.

#### 0.2.1
- Introduce `relax` feature to `has_schema` for pandas dataframes.

#### 0.2.0
- Rename `input_has_schema` to `has_schema` and add output dataframe check.

#### 0.1.38
- Fix Rmd conda env.

#### 0.1.37
- Update in test_requirements.in file


#### 0.1.36
- Fix doc.


#### 0.1.35
- Add `fill_nulls`

#### 0.1.34
- Update docs.

#### 0.1.33
- Fix `inject` to enable instantiating objects without arguments.

#### 0.1.32
- Add `add_input_kwarg_filter`

#### 0.1.31
- Make `make_list_regexable` more robust. 

#### 0.1.30
- Fix bug in `input_has_schema`.

#### 0.1.29
- Add exclude keys functionality to `inject_object()`

#### 0.1.28
- Update docs. 

#### 0.1.27
- Update `make_list_regexable` to be explicitly turned on.

#### 0.1.26

- Update requirements so that `pyspark` is explicitly below `<4.0`
#### 0.1.25
- More docs fixes.

#### 0.1.24
- Updated `make_regexable` decorator to be able to raise an exception.

#### 0.1.23
- Fix docs.

#### 0.1.22
- Removed `node` decorator.

#### 0.1.21
- Updated individual decorators documentation for `make_regexable` decorator.

#### 0.1.20
- Added `augment` decorator.

#### 0.1.19
- Update column name in `primary_key` decorator.

#### 0.1.18
- Removed useless suppression from `output_primary_key` decorator.

#### 0.1.17
- Added `make_list_regexable` decorator.

#### 0.1.16
- Cleaned up `requirements.txt`.

#### 0.1.15
- Suppressed `consider-using-f-string` pylint message after the version upgrade to `pylint-2.11.1`.

#### 0.1.14
- Added `defender primary_key_check` decorator.

#### 0.1.13
- Added `output_primary_key` decorator.

#### 0.1.12
- Added `add_output_filter` decorator.

#### 0.1.11
- Renamed from `requirements.in` to `requirements.txt`.

#### 0.1.10
- Updated brix post namespace.

#### 0.1.9
- Added `BuiltinFunctionType` check in refit while instantiating object.

#### 0.1.8
- Updated pytest fixture scope for `dummy_pd_df` and `dummy_spark_df`.

#### 0.1.7
- Added versioning(v1) to refit.

#### 0.1.6
- Fix useless pylint disables.

#### 0.1.5
- Updated README for individual decorators.

#### 0.1.4
- Updated README for individual decorators.

#### 0.1.3
- Added wrapper functions and tests for missing decorators.

#### 0.1.2
- Added README to showcase individual decorator: `defender` style `has_schema`.

#### 0.1.1
- Added individual decorator for `inject_object`.
- Added README to showcase individual decorators: `inject_object`.

#### 0.1.0
- Added decorators:
    - `node` which contains `_inject_object`, `_retry`, `_input_has_schema_node`, `_add_input_kwarg_filter` and `_unpack_params`.
    - `input_kwarg_filter`.
    - `input_has_schema`.
    - Defender style `has_schema`.
