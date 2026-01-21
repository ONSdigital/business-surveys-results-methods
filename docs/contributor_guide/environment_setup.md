# Detailed instructions on setting up Python and installing `uv` on ONS PCs

## Setting up Python on ONS PCs
Make sure you have git and Python installed on your computer (using Service Now) and set up python and pip using these instructions [ASAP wiki](https://gitlab-app-l-01/ASAP/coding-getting-started-guide/-/wikis/python).

It is useful to follow the ASAP instructions above to create a `.bat` file that will enable you to access that version of Python using with a shortcut, such as `python3_11`. If you haven't created this .bat file, you will need to replace `python3_11` with the path to where you have python 3.11 installed, for example `"C:\ONSapps\My_Python\Python_3_11.python.exe"`

## 1. Cloning this repo

Open your terminal in the folder you want to save your repo in. Note that you don't need to create a folder for the name of the repo, cloning will automatically do this. Input the following line into your terminal:

git clone https://github.com/ONSdigital/business-surveys-results-methods.git

## 2. Installing uv for the virtual environment and package management

### 2.1 Preparing to use `uv` and `pipx`

This section can be skipped if you have performed these steps for a different project.

In order to install uv on your Windows machine so it is not restricted to a particular environment, use pipx.
To learn more about `pipx`, see the [technical documentation](uv_and_pipx.md).

You may first need to install pipx:

```
python3_11 -m pip install pipx
```
To ensure your environment PATH is updated, run the following:
```
python3_11 -m pipx ensurepath
```
After this you will need to restart your terminal.

You are now ready to install uv:

To learn more about `uv`, see the [technical documentation](https://github.com/ONSdigital/iabs-results-processing/blob/iabs_19_uv_docs/docs/uv_and_pipx.md).
```
python3_11 -m pipx install uv
```
and again update your path:
```
python3_11 -m pipx ensurepath
```
Again, restart your terminal, and now `uv` should be available globally - check this by running `uv --version`

There is one more step to ensure uv will use the ONS Aritfactory and the ONSApps version of Python.

Access 'Roaming' folder by pasting the following path in the navigation bar in Windows Explorer or by navigating to:

```
`C:\\Users<windows_username>\AppData\Roaming`
```
Create a folder named 'uv' inside Roaming directory:

```
`C:\\Users<windows_username>\AppData\Roaming\uv`
```

inside the uv folder, create a file named uv.toml and paste in the following.

replacing <username> and <encr_password> with your windows username and encripted password respectively.

```
cache-dir = "D:/uv/cache"

allow-insecure-host = ["onsart-01"]
python-preference = "only-system"
python-downloads = "never"

[[index]]
url = "https://<username>:<encr_password>@onsart-01/artifactory/api/pypi/yr-python/simple"
default = true

```
### 2.2 Installing global developer tools (ruff, black, pre-commit)
These tools support code quality, secure commits, and formatting. They should be installed globally using pipx so they are available across all projects and don't interfere project's virtual environments.

To install these tools, run:
``` sh
python3_11 -m  pipx install ruff
python3_11 -m  pipx install black
python3_11 -m  pipx install pre-commit
```

## 3. Creating a virtual environment and installing packages using uv
To create a virtual environment, simply open a terminal and type

```sh
uv venv --python="C:\ONSapps\My_Python\Python_3_11"
```
To activate the environment, run this:
```sh
.venv\Scripts\activate
```
You will now have a new folder named .venv at the root level of your repo to contain the version of Python and all other items to install for this project. You should also see `(iabs-results-processing)` at the beginning of each line in your terminal.

Finally, you are now able to install all packages using the folling command:

```sh
uv sync --all-extras
```

NOTE: `uv` will use `pyproject.toml` for a list of dependancies. The option `--all-extras` is included as we want to include the packages in the `dependency_groups` section.

## Pre-commit actions
This repository contains a configuration of pre-commit hooks. These are language agnostic and focussed on repository security and coding style. If approaching this project as a developer, you should install and enable pre-commits by running the following in your shell:
```sh
pre-commit install
```
Once pre-commits are activated, whenever you commit to this repository a series of checks will be executed. If any of these checks fail, the commit will be rejected and you will be prompted to fix the issues, stage the files and commit again.
