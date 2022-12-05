import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("user")
args = parser.parse_args()


def run():
    user = args.user
    assert len(user) > 0, 'Enter valid username'
    djimaging_dir = os.path.split(os.path.abspath(__file__))[0]
    user_dir = create_user_dir(djimaging_dir, user)
    copy_notebooks(djimaging_dir, user_dir, user)
    copy_table(djimaging_dir, user_dir)
    copy_schema(djimaging_dir, user_dir, user)


def create_user_dir(djimaging_dir, user):
    assert os.path.isdir(os.path.join(djimaging_dir, 'djimaging'))
    all_user_dir = os.path.join(djimaging_dir, 'djimaging', 'user')
    make_dir_if_new(all_user_dir)  # Make if not exists
    open(os.path.join(all_user_dir, '__init__.py'), 'a').close()
    user_dir = os.path.join(all_user_dir, user)
    make_dir_if_new(user_dir)
    open(os.path.join(user_dir, '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'schemas'))
    open(os.path.join(user_dir, 'schemas', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'tables'))
    open(os.path.join(user_dir, 'tables', '__init__.py'), 'a').close()
    make_dir_if_new(os.path.join(user_dir, 'notebooks'))

    return user_dir


def replace_schema_name(filepath, username):
    # Read in the file
    with open(filepath, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('djimaging.schemas.advanced_schema', f'djimaging.user.{username}.schemas.my_schema')

    # Write the file out again
    with open(filepath, 'w') as file:
        file.write(filedata)


def copy_notebooks(djimaging_dir, user_dir, username):
    files = os.listdir(os.path.join(djimaging_dir, 'djimaging/notebooks'))
    notebooks = [nb for nb in files
                 if os.path.isfile(os.path.join(djimaging_dir, 'djimaging/notebooks', nb)) and nb.endswith('.ipynb')]
    for nb in notebooks:
        src = os.path.join(djimaging_dir, 'djimaging/notebooks', nb)
        dst = os.path.join(user_dir, 'notebooks', nb)

        if not os.path.isfile(dst):
            shutil.copy(src, dst)
            replace_schema_name(dst, username)


def copy_schema(djimaging_dir, user_dir, username):
    src = os.path.join(djimaging_dir, 'djimaging/schemas/_new_schema_example.py')
    dst = os.path.join(user_dir, 'schemas/my_schema.py')

    if not os.path.isfile(dst):
        shutil.copy(src, dst)

        with open(os.path.join(user_dir, 'schemas/my_schema.py'), "r+") as f:
            text = f.read()
            f.seek(0)

            new_import = \
                f"# Remove comments to import templates that listed in user/{username}/tables/__init__.py \n" + \
                f"# from djimaging.user.{username}.tables import *"

            new_table = \
                "\n" + \
                "# Remove comments to add table to schema\n" + \
                "# @schema\n" + \
                "# class ExampleTable(ExampleTableTemplate):\n" + \
                "#     field_table = Field\n"

            f.write(new_import + "\n" + text + "\n" + new_table)


def copy_table(djimaging_dir, user_dir):
    src = os.path.join(djimaging_dir, 'djimaging/tables/_new_table_example.py')
    dst = os.path.join(user_dir, 'tables/my_table.py')

    if not os.path.isfile(dst):
        shutil.copy(src, dst)

        with open(os.path.join(user_dir, 'tables/__init__.py'), "a") as f:
            f.write("from .my_table import ExampleTableTemplate\n")


def make_dir_if_new(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    run()
