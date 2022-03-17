import os
import argparse
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
    make_dir_if_new(os.path.join(djimaging_dir, 'djimaging', 'user'))  # Make if not exists

    user_dir = os.path.join(djimaging_dir, 'djimaging', 'user', user)
    make_dir_if_new(user_dir)
    make_dir_if_new(os.path.join(user_dir, 'schemas'))
    make_dir_if_new(os.path.join(user_dir, 'tables'))
    make_dir_if_new(os.path.join(user_dir, 'notebooks'))

    return user_dir


def copy_notebooks(djimaging_dir, user_dir):
    files = os.listdir(os.path.join(djimaging_dir, 'notebooks'))
    notebooks = [nb for nb in files
                 if os.path.isfile(os.path.join(djimaging_dir, 'notebooks', nb)) and nb.endswith('.ipynb')]
    for nb in notebooks:
        shutil.copy(src=os.path.join(djimaging_dir, 'notebooks', nb), dst=os.path.join(user_dir, 'notebooks', nb))


def make_dir_if_new(path):
    if not os.path.exists(path):
        os.mkdir(path)


run()
