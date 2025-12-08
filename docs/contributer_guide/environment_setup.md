# Detailed instructions on setting up Python and installing `uv` on ONS machines

## Setting up Python on ONS PCs
Make sure you have git and Python installed on your computer (using Service Now) and set up python and pip using these instructions [ASAP wiki](https://gitlab-app-l-01/ASAP/coding-getting-started-guide/-/wikis/python).

It is useful to follow the ASAP instructions above to create a `.bat` file that will enable you to access that version of Python using with a shortcut, such as `python3_11`. You can then use that to replace
`<path_to_python>` below.

NOTE: As developers we are using Python3.11 as this is the latest version available in the DAP environment.

## 2. Installing uv for the virtual environment and package management
To learn more about `pipx`, see the [technical details document](docs/setup_technical_details.md#3-what-pipx-is-simple-explanation).
This section can be skipped if you have performed it for a different project.

In order to install uv on your Windows machine so it is not restricted to a particular
environment, use pipx. You may first need to install pipx:
```
<path_to_python> -m pip install pipx
```
To ensure your environment PATH is updated, run the following:
```
pipx ensurepath
```
After this you will need to restart your terminal.

You are now ready to install uv:

To learn more about uv, see the [technical details document](docs/setup_technical_details.md#4-what-uv-is-simple-explanation).
```
pipx install uv
```
and again update your path:
```
pipx ensurepath
```
Again, restart your terminal, and now `uv` will be available globally.

There is one more step to ensure `uv` will use the ONS Artifactory and the ONSApps version of Python. Create a toml file in this location:

```
%APPDATA%\uv\uv.toml
```
`%APPDATA%` can be accessed by pasting in the navigation bar in Windows Explorer or by navigating to `C:\\Users\<windows_username>\AppData\Roaming`

Paste in the following, replacing `<username>` and `<encr_password>` with your windows username and encrypted password respectively.

```
cache-dir = "D:/uv/cache"

allow-insecure-host = ["onsart-01"]
python-preference = "only-system"
python-downloads = "never"

[[index]]
url = "https://<username>:<encr_password>@onsart-01/artifactory/api/pypi/yr-python/simple"
default = true
```
## 2.1 Installing global developer tools (ruff, black, pre-commit)
These tools support code quality, secure commits, and formatting. They should be installed globally using pipx so they are available across all projects and don't interfere project's virtual environments.

To install these tools, run:
```
pipx install ruff
pipx install black
pipx install pre-commit
```
## 3. Setting up a virtual environment using `uv` and the ONSApps Python version

To ensure the environment uses Python 3.11, run:
```sh
uv venv --python="C:\ONSapps\My_Python\Python_3_11"
```
(Replace with the path to where you have python installed, if necessary)
Next, activate the environment:
```sh
.venv\Scripts\activate
```

## 4. Installing pre-commits

For security and consistency of coding best practice, please set up pre-commit to run when you commit work:
```sh
pre-commit install
```

## 5. Install the package in editable mode using uv for efficiency

`uv pip install -e .`
