![Build status](https://github.com/eulerlab/djimaging/actions/workflows/python-app.yml/badge.svg)

# djimaging

2P imaging data joint tables and schemas

## Getting started

### Create a MySQL account

Ask IT to create a MySQL user account for you. They will send you a username and activation password.
> ❗ You probably don't want your own local MySQL server, so don't try to set one up but get access to the shared one.

### Create a remote docker container

Create a docker container and install this package and its requirements in it.

Download the package:

```bash
git clone https://github.com/eulerlab/djimaging.git
````

Install the package e.g. using pip as an editable package:

```bash
cd djimaging
pip install -r requirements.txt
pip install -e .
```

If you want to use autorois, also install the following:
```
pip install -r requirements-autorois.txt
```

> ❗ To test if the package was successfully installed, e.g.
> open a jupyter notebook in your container and call <code>import djimaging</code>.

### Create a config file

Create a datajoint config <code>*.json</code> file,
e.g. based on the template <code>djimaging/djconfig/djconf_template.json</code>.
Fill out the missing values; if you don't know how, ask someone in your group.
> ❗ Never upload this personal config file to GitHub.

### Create a user folder

Inside the root folder <code>djimaging</code> (not in <code>djimaging/djimaging</code>)
run the make user script my calling

```bash
python3 make_user.py your_username_here
```

This will create a folder <code>djimaging/djimaging/user/your_username_here/notebooks</code>
with some tutorial notebooks.
> ❗ Per default everything in this folder will not be under version control.
> Consider adding an expectation for your files here, but do not upload them to the shared repository.
> Do not upload personal config files.

### Clean up

When you no longer need your test schema, make sure you drop it by
calling <code>schema.drop()</code> and confirm by entering <code>yes</code>.

> ⚠️ Make sure you only drop your own schema! <code>schema.drop()</code> will show you the name of the schema.
> If you are not sure about the schema's origin, don't drop it!
