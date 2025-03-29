# Configuration Files Setup Instructions

To properly organize configuration files, you need to ensure they're in the correct directory structure. Here's how to set up the configuration files:

## Directory Structure
Create the following directory structure in your project root:

```
src/
  glaucoma_detection/
    ...
conf/
  config.yaml               # Main config file
  data/
    default.yaml
  model/
    unet.yaml
  preprocessing/ 
    default.yaml
  training/
    default.yaml
  evaluation/
    default.yaml
    model/
      unet.yaml
  logging/
    default.yaml
```

## Move Configuration Files
Move the existing configuration files to their proper locations:

1. Move `data/default.yaml` to `conf/data/default.yaml`
2. Move `training/default.yaml` to `conf/training/default.yaml`
3. Move `model/unet.yaml` to `conf/model/unet.yaml`
4. Move `config.yaml` to `conf/config.yaml`

## Update Import Paths
Make sure your imports in Python files point to the correct locations. For example:

```python
@hydra.main(config_path="../conf", config_name="config", version_base=None)
```

The `config_path` should be a relative path from your Python file to the `conf` directory.

## Configuration Loading with Hydra
Hydra will automatically merge configuration files based on the defaults specified in your main `config.yaml` file. The structure of `config.yaml` should look like:

```yaml
# @package _global_
defaults:
  - data: default
  - model: unet
  - preprocessing: default
  - training: default
  - evaluation: default
  - logging: default
  - _self_

# Other global configurations...
```

This tells Hydra to load and merge these configuration files in the specified order.