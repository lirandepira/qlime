# Description

# User guide

# Administration Guide
## Bump version

Bumping a version requires bumpver.
If bumpver is not installed yet use

```Shell
pip install bumpver
```

### Bump Z in X.Y.Z

```Shell
bumpver upgrade -patch
```

### Bump Y in X.Y.Z

```Shell
bumpver upgrade -minor
```

### Bump X in X.Y.Z

```Shell
bumpver upgrade -major
```

## Publish to the TEST Pypi repository

Once the version number has been properly bumped use (note your credentials must be properly setup into your `$HOME/.pypirc` file:
```Shell
rm -rf dist/
python -m build
twine upload -r testpypi dist/*
```
