# Remote GPU Execution 

## Local Execution 
```
kedro run --node data_bowl.trained_model
```
## Remote Execution
```
kedro run --node data_bowl.trained_model --runner runner.RayRunner --env ray
```