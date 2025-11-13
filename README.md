# chemprop-contrib

Community-contributed extensions to the Chemprop machine learning package

# Contributing

All community-contributed packages should fork the repository and in a branch on the fork, make a new directory under the chemprop_contrib directory with their name. Then add the following:
 - License for the code (preferably MIT)
 - All associated code files
 - Data files, if needed, and their provenance
 - Documentation (perhaps a Notebook (*.ipynb) or README.md) with instructions on how to use code contributed
 - Tests demonstrating that the code works as intended

 If your package has additional dependencies, add them to the list of optional dependencies in the pyproject.toml with the same name as your directory.

Once the package is ready, make a pull request to the main chemprop-contrib repository for a brief review.
We provide a template for these pull requests.

## Examples

For packages which _do not_ require any new dependencies, see `moe_regressor` as an example.

For packages whi _do_ require additional dependencies, see `mcp` as an example.
Note that one must edit the `pyproject.toml` and _add_ these dependencies, as mentioned in [Contributing](#contributing).

# Using

chemprop-contrib can either be pip installed directly
```
pip install chemprop-contrib
```
or installed as an optional dependency when installing Chemprop
```
pip install chemprop[contrib]
```

Then, the contributed packages can be imported in Python as
```python
from chemprop_contrib import <package_name>
```

Each package includes tests, which can be used to demonstrate the use of the package.

# Available Packages

 - `moe_regressor`: Implements the ["Adaptive Mixture of Local Experts"](https://doi.org/10.1162/neco.1991.3.1.79) model for regression.
