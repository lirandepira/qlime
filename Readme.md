# Description

# Usage guide

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

Once the version has been bumped use:
```Shell
rm -rf dist/
python -m build
twine upload -r testpypi dist/*
```
