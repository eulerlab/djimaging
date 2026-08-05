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

Install the package as an editable package using [uv](https://docs.astral.sh/uv/):

```bash
cd djimaging
uv pip install -e .
```

If you want to use autorois or receptive fields, install with the optional extras:
```bash
uv pip install -e ".[autorois]"
uv pip install -e ".[rf]"
uv pip install -e ".[autorois,rf]"
```

> ❗ To test if the package was successfully installed, e.g.
> open a jupyter notebook in your container and call <code>import djimaging</code>.

### Create a config file

Create a datajoint config <code>*.json</code> file,
e.g. based on the template <code>djimaging/djconfig/djconf_template.json</code>.
Fill out the missing values; if you don't know how, ask someone in your group.
> ❗ Never upload this personal config file to GitHub.

DataJoint 2 requires MySQL 8 with `utf8mb4`/`utf8mb4_bin` and named stores.
Use separate roots for externally managed acquisition files, processed arrays
and blobs, and model attachments. Paths inserted into `<filepath@...>` fields
must be relative to their configured store root.

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

## Local testing

Install the test environment with DataJoint 2.3.x:

```bash
uv pip install pytest "setuptools<81" -e .
```

Run unit tests:

```bash
pytest -q tests --ignore=tests/integration
```

Run integration tests against a temporary MySQL 8.0.43 server:

```bash
docker run --rm --name djimaging-mysql-test \
  -e MYSQL_ROOT_PASSWORD=datajoint -e MYSQL_ROOT_HOST=% \
  -p 3307:3306 -d mysql:8.0.43 \
  --character-set-server=utf8mb4 --collation-server=utf8mb4_bin
until docker exec djimaging-mysql-test mysqladmin ping -h 127.0.0.1 -pdatajoint; do sleep 1; done

DJ_TEST_MYSQL=1 DJ_HOST=127.0.0.1 DJ_PORT=3307 \
DJ_USER=root DJ_PASS=datajoint pytest -q tests/integration

docker stop djimaging-mysql-test
```
