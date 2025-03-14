# Example model files

## You custom model

`custom_model_example.json` contains a new model which is not present in `enterprise_warp`. It is `my_powerlaw`, the name is then matched with class methods of your custom model class. In particular, we have set up `../run_example_paramfile.py` to load the custom model class `CustomModels` from `../custom_models.py`. Thus class has a method `def my_powerlaw`. Thus, `my_powerlaw` from the `.json` noise model file will be matched to this method.

## Spin noise model selection

1. `default_noise_example_1.json`: power law spin noise.
2. `default_noise_example_2.json`: power law with a low-frequency turnover.
