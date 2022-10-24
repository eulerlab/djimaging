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
    copy_notebooks(djimaging_dir, user_dir)


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
    make_dir_if_new(os.path.join(user_dir, 'djimaging/notebooks'))

    return user_dir


def copy_notebooks(djimaging_dir, user_dir):
    files = os.listdir(os.path.join(djimaging_dir, 'djimaging/notebooks'))
    notebooks = [nb for nb in files
                 if os.path.isfile(os.path.join(djimaging_dir, 'djimaging/notebooks', nb)) and nb.endswith('.ipynb')]
    for nb in notebooks:
        src = os.path.join(djimaging_dir, 'djimaging/notebooks', nb)
        dst = os.path.join(user_dir, 'djimaging/notebooks', nb)

        if not os.path.isfile(dst):
            shutil.copy(src, dst)


def make_dir_if_new(path):
    if not os.path.exists(path):
        os.mkdir(path)


run()
