# Overview

Thanks for taking the time to contribute! We appreciate all contributions, from reporting bugs to
implementing new features. If you're unclear on how to proceed after reading this guide, please
open a [new discussion](https://github.com/E-MAB/GMAB/discussions/new/choose).

## Reporting bugs

We use [GitHub issues](https://github.com/E-MAB/GMAB/issues) to track bugs. 
You can report a bug by opening a [new issue](https://github.com/E-MAB/GMAB/issues/new/choose).

Before creating a bug report, please check that your bug has not already been reported, and that
your bug exists on the latest version of GMAB. If you find a closed issue that seems to report the
same bug you're experiencing, open a new issue and include a link to the original issue in your
issue description.

Please include as many details as possible in your bug report. The information helps the maintainers
resolve the issue faster.

## Suggesting enhancements

We use [GitHub Discussions](https://github.com/E-MAB/GMAB/discussions) to track suggested
enhancements. You can suggest an enhancement by opening a
[new feature request](https://github.com/E-MAB/GMAB/discussions/new/choose).
Before creating an enhancement suggestion, please check that a similar issue does not already exist.

Please describe the behavior you want and why, and provide examples of how gmab would be used if
your feature were added.

## Contributing to the codebase

### Picking an issue

Pick an issue by going through the [issue tracker](https://github.com/E-MAB/GMAB/issues) and
finding an issue you would like to work on.

If you would like to take on an issue, please comment on the issue to let others know. You may use
the issue to discuss possible solutions.

### Setting up your local environment

The GMAB development flow relies on both Rust and Python, which means setting up your local
development environment is not trivial. If you run into problems, please open a
[new discussion](https://github.com/E-MAB/GMAB/discussions/new/choose)

#### Configuring Git

For contributing to GMAB you need a free [GitHub account](https://github.com) and have
[git](https://git-scm.com) installed on your machine. Start by
[forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the GMAB repository, then
clone your forked repository using `git`:

```bash
git clone https://github.com/<username>/GMAB.git
cd GMAB
```

Optionally set the `upstream` remote to be able to sync your fork with the GMAB repository in the
future:

```bash
git remote add upstream https://github.com/E-MAB/GMAB.git
git fetch upstream
```

#### Installing dependencies

In order to work on GMAB effectively, you will need [Rust](https://www.rust-lang.org/) and
[Python](https://www.python.org/).

First, install Rust using [rustup](https://www.rust-lang.org/tools/install).

Next, install Python, for example using [uv](https://docs.astral.sh/uv/getting-started/installation/).

You can now check that everything works correctly by going into the `pygmab` directory and
running the test suite (warning: this may be slow the first time you run it):

```bash
cd pygmab
uv tool install maturin
maturin develop
uv run pytest
```

This will do a number of things:

- Use Python to create a virtual environment in the `.venv` folder.
- Use [uv](https://github.com/astral-sh/uv) to install all Python
  dependencies for development, linting, and building documentation.
- Use Rust to compile and install GMAB in your virtual environment.
- Use [pytest](https://docs.pytest.org/) to run the Python unittests in your virtual environment

Check if linting also works correctly by running:

```bash
pre-commit
```

If this all runs correctly, you're ready to start contributing to the GMAB codebase!

#### Updating the development environment

Dependencies are updated regularly. If you do not keep your environment
up-to-date, you may notice tests or CI checks failing, or you may not be able to build GMAB at
all.

To update your environment, first make sure your fork is in sync with the GMAB repository:

```bash
git checkout main
git fetch upstream
git rebase upstream/main
git push origin main
```

Update all Python dependencies to their current versions in origin/main by running:

```bash
uv sync
```

If the Rust toolchain version has been updated, you should update your Rust toolchain. Follow it up
by running `cargo clean` to make sure your Cargo folder does not grow too large:

```bash
rustup update
cargo clean
```

### Working on your issue

Create a new git branch from the `main` branch in your local repository, and start coding!

The Rust code is located in the `gmab` directory, while the Python codebase is located in the
`pygmab` directory. 

Two other things to keep in mind:

- If you add code that should be tested, add tests.
- If you change the public API, update the documentation.

### Pull requests

When you have resolved your issue,
[open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
in the GMAB repository. Please adhere to the following guidelines:

- Title:
    - Start your pull request title with a [conventional commit](https://www.conventionalcommits.org/) tag.
      This helps us add your contribution to the right section of the changelog.
      We use the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type).
      Scope can be `rust` and/or `python`, depending on your contribution: this tag determines which changelog(s) will include your change.
      Omit the scope if your change affects both Rust and Python.
    - Use a descriptive title starting with an uppercase letter.
      This text will end up in the [changelog](https://github.com/E-MAB/GMAB/releases), so make sure the text is meaningful to the user.
      Use single backticks to annotate code snippets.
      Use active language and do not end your title with punctuation.
    - Example: ``fix(python): Fix `DataFrame.top_k` not handling nulls correctly``
- Description:
    - In the pull request description, [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) to the issue you were working on.
    - Add any relevant information to the description that you think may help the maintainers review your code.
- Make sure your branch is [rebased](https://docs.github.com/en/get-started/using-git/about-git-rebase) against the latest version of the `main` branch.
- Make sure all GitHub Actions checks pass.


After you have opened your pull request, a maintainer will review it and possibly leave some
comments. Once all issues are resolved, the maintainer will merge your pull request, and your work
will be part of the next GMAB release!

Keep in mind that your work does not have to be perfect right away! If you are stuck or unsure about
your solution, feel free to open a draft pull request and ask for help.

## Contributing to documentation

The most important components of GMAB documentation are the
[user guide](https://example.com/), the
[API references](https://example.com/).

### User guide

The user guide is maintained in the `tbd` folder. Before creating a PR first
raise an issue to discuss what you feel is missing or could be improved.

#### Building and serving the user guide

The user guide is built using [MkDocs](https://www.mkdocs.org/).

Activate the virtual environment and run `mkdocs serve` to build and serve the user guide, so you
can view it locally and see updates as you make changes.

#### Creating a new user guide page

Each user guide page is based on a `.md` markdown file. This file must be listed in `mkdocs.yml`.

#### Adding a shell code block

To add a code block with code to be run in a shell with tabs for Python and Rust, use the following
format:

````
=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install fsspec
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    $ cargo add aws_sdk_s3
    ```
````

#### Adding a code block

The snippets for Python and Rust code blocks are in the `docs/source/src/python/` and
`docs/source/src/rust/` directories, respectively. To add a code snippet with Python or Rust code to
a `.md` page, use the following format:

```
{{code_block('user-guide/io/cloud-storage','read_parquet',['read_parquet','read_csv'])}}
```

- The first argument is a path to either or both files called
  `docs/source/src/python/user-guide/io/cloud-storage.py` and
  `docs/source/src/rust/user-guide/io/cloud-storage.rs`.
- The second argument is the name given at the start and end of each snippet in the `.py` or `.rs`
  file
- The third argument is a list of links to functions in the API docs. For each element of the list
  there must be a corresponding entry in `docs/source/_build/API_REFERENCE_LINKS.yml`

If the corresponding `.py` and `.rs` snippet files both exist then each snippet named in the second
argument to `code_block` above must exist or the build will fail. An empty snippet should be added
to the `.py` or `.rs` file if the snippet is not needed.

Each snippet is formatted as follows:

```python
# --8<-- [start:read_parquet]
import gmab
# --8<-- [end:read_parquet]
```

The snippet is delimited by `--8<-- [start:<snippet_name>]` and `--8<-- [end:<snippet_name>]`. The
snippet name must match the name given in the second argument to `code_block` above.

In some cases, you may need to add links to different functions for the Python and Rust APIs. When
that is the case, you can use the two extra optional arguments that `code_block` accepts, that can
be used to pass Python-only and Rust-only links:

```
{{code_block('path', 'snippet_name', ['common_api_links'], ['python_only_links'], ['rust_only_links'])}}
```

#### Linting

Before committing, install `pre-commit install` (see above)

### API reference

GMAB has separate API references for [Rust](https://example.com/) and
[Python](https://example.com/). These are generated directly
from the codebase, so in order to contribute, you will have to follow the steps outlined in
[this section](#contributing-to-the-codebase) above.

#### Rust

Rust GMAB uses `cargo doc` to build its documentation. Contributions to improve or clarify the API
reference are welcome.

#### Python

For the Python API reference, we always welcome good docstring examples. This is a great way to start contributing to GMAB!

Note that we follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)
convention. Docstring examples should also follow the [Black](https://black.readthedocs.io/)
codestyle.

GMAB uses Sphinx to build the API reference. This means docstrings in general should follow the
[reST](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) format.

## Release flow

_This section is intended for GMAB maintainers._

GMAB releases Rust crates to [crates.io](https://crates.io/crates/gmab) and Python packages to
[PyPI](https://pypi.org/project/gmab/).

New releases are marked by an official [GitHub release](https://github.com/E-MAB/GMAB/releases)
and an associated git tag. We utilize
[Release Drafter](https://github.com/release-drafter/release-drafter) to automatically draft GitHub
releases with release notes.

### Steps

The steps for releasing a new Rust or Python version are similar. The release process is mostly
automated through GitHub Actions, but some manual steps are required. Follow the steps below to
release a new version.

Start by bumping the version number in the source code:

1. Check the [releases page](https://github.com/E-MAB/GMAB/releases) on GitHub and find the
   appropriate draft release. Note the version number associated with this release.
2. Make sure your fork is up-to-date with the latest version of the main GMAB repository, and
   create a new branch.
3. Bump the version number.

- _Rust:_ Update the version number in all `Cargo.toml` files in the `gmab` directory and
  subdirectories. You'll probably want to use some search/replace strategy, as there are quite a few
  crates that need to be updated.
- _Python:_ Update the version number in
  [`pygmab/Cargo.toml`](https://github.com/E-MAB/GMAB/blob/main/pygmab/Cargo.toml#L3) to
  match the version of the draft release.

4. From the `pygmab` directory, run `cargo build` to generate a new `Cargo.lock` file.
5. Create a new commit with all files added. The name of the commit should follow the format
   `release(<language>): <Language> GMAB <version-number>`. For example:
   `release(python): Python GMAB 0.16.1`
6. Push your branch and open a new pull request to the `main` branch of the main GMAB repository.
7. Wait for the GitHub Actions checks to pass, then squash and merge your pull request.

Directly after merging your pull request, release the new version:

8. Go to the release workflow
   ([Python](https://github.com/E-MAB/GMAB/actions/workflows/release-python.yml)/[Rust](https://github.com/E-MAB/GMAB/actions/workflows/release-rust.yml)),
   click _Run workflow_ in the top right, and click the green button. This will trigger the
   workflow, which will build all release artifacts and publish them.
9. Wait for the workflow to finish, then check
   [crates.io](https://crates.io/crates/gmab)/[PyPI](https://pypi.org/project/gmab/)/[GitHub](https://github.com/E-MAB/GMAB/releases)
   to verify that the new GMAB release is now available.

### Troubleshooting

It may happen that one or multiple release jobs fail. If so, you should first try to simply re-run
the failed jobs from the GitHub Actions UI.

If that doesn't help, you will have to figure out what's wrong and commit a fix. Once your fix has
made it to the `main` branch, simply re-trigger the release workflow.
