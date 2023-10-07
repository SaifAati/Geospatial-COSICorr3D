
## Transitioning to `poetry` for Python Packaging and Dependency Management

`poetry` offers a more streamlined and modern approach to packaging and dependency management in Python. 

### 1. Install Poetry


```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Initialize Poetry

Navigate to your project's root directory and run:

```bash
poetry init
```

This command will guide you through creating a `pyproject.toml` for your project. You can accept the defaults or modify them as needed.

### 3. Add Dependencies

Instead of manually adding dependencies to the `pyproject.toml`, you can use poetry commands to add them. For example:

```bash
poetry add numpy
```

This will add numpy as a dependency and update the `pyproject.toml` and `poetry.lock` files.

### 4. Specify Your Package

In the `pyproject.toml`, under `[tool.poetry]`, you will need to specify your package details. This will include the name, version, description, and other metadata. You can also specify the package's source directory if it's not in the root.

### 5. Move Other Configurations

You can move configurations for tools like `pytest` and `mypy` from your existing `pyproject.toml` to the one managed by `poetry`.

### 6. Building and Publishing

With `poetry`, building and publishing your package becomes straightforward. To build:

```bash
poetry build
```

And to publish to PyPI:

```bash
poetry publish
```

### 7. Local Development

For local development, `poetry` provides a virtual environment to ensure dependencies don't clash with global packages. Use:

```bash
poetry shell
```

to activate this environment.

### 8. Remove `setup.py` and `setup.cfg`

Once you've fully transitioned to `poetry`, you can safely remove `setup.py` and `setup.cfg` as they are no longer needed.
