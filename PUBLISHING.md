# Publishing fleet-caliper

The distribution is built and validated (`twine check` passes; clean-install in a
fresh zero-dependency venv imports the full public API). These are the steps to
publish. Tokens are secrets — they live only on your machine, never in this repo.

## One-time: tokens

- **PyPI token:** https://pypi.org/manage/account/token/ — scope "Entire account"
  for the first upload of a brand-new project (rotate to project-scoped after).
- **TestPyPI token (for the rehearsal):** https://test.pypi.org/manage/account/token/

Then either paste tokens at the prompt, or set up `~/.pypirc` once:

```bash
cp ~/.pypirc.template ~/.pypirc   # template created alongside this repo's setup
# edit ~/.pypirc, paste the pypi-… tokens
chmod 600 ~/.pypirc
```

## Step 1 — rehearse on TestPyPI (recommended)

Real PyPI will not let you re-upload a version. Rehearse first so a typo costs
nothing:

```bash
cd /Users/vasundra.srinivasan/Desktop/fleet-caliper
.venv/bin/python -m build                       # rebuild if anything changed
.venv/bin/twine check dist/*
.venv/bin/twine upload --repository testpypi dist/*
```

Verify the rehearsal (TestPyPI needs PyPI as an extra index for any real deps):

```bash
python3 -m venv /tmp/verify-test
/tmp/verify-test/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  fleet-caliper
/tmp/verify-test/bin/python -c "import caliper; print(caliper.__version__)"
```

## Step 2 — publish to real PyPI

```bash
cd /Users/vasundra.srinivasan/Desktop/fleet-caliper
.venv/bin/twine upload dist/*
```

If not using `~/.pypirc`: username is the literal `__token__`, password is your
`pypi-…` token.

## Step 3 — verify the real release

```bash
python3 -m venv /tmp/verify
/tmp/verify/bin/pip install fleet-caliper
/tmp/verify/bin/python -c "import caliper; print('published', caliper.__version__)"
```

## Releasing a new version later

1. Bump `version` in `pyproject.toml` and `__version__` in `src/caliper/__init__.py`.
2. `rm -rf dist && .venv/bin/python -m build`
3. `.venv/bin/twine upload dist/*`
4. Tag it: `git tag v0.1.x && git push --tags`
