# Isolation Forest

I've implemented the algorithm using the papers inside `papers` directory.
These are the initial papers that invented and described isolation forest algorithm.

## Project structure

`isolation_forest.py` - There is the Python implementation of the algorithm.

`data.csv` - The data that I used for algorithm implementation and testing.

`runner.py` and `runner.ipynb` - These are the files that I used for testing and debugging.

`sklearn_implementation.ipynb` - This is the implementation of the algorithm using scikit-learn library.
I used this to compare my results to the results of the scikit-learn implementation.

## How to run

If you want to debug the algorithm to check each step and logic then use `runner.py`

If you want the see the results of the algorithm, how it detects anomalies, plus the
visualization of the data then use `runner.ipynb`

If you want to check if my implementation is correct, considering the randomness inside the model,
then use `sklearn_implementation.ipynb` and compare the results.
