# Indecision Modeling 
Codebase accompanying _Indecision Modeling_ (McElfresh et al., AAAI'20).

### Dependencies
- numpy
- scipy
- pandas
- [ax](https://ax.dev/)

### Code

- `preference_classes.py`: classes for modeling preferences
- `fit_preference_models.py`: functions for fitting model prameters using random search
- `run_test*.py`: experiments for fitting various models to observed data
- `utils.py`: various helper functions

### ./data

This contains all cleaned data used in our experiments, see `./data/README.txt` for details.