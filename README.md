![Build status](https://github.com/jonathanoesterle/djimaging/actions/workflows/python-app.yml/badge.svg)

# djimaging
2P imaging data joint tables and schemas

## Installation
Download the package:
```bash
git clone https://github.com/jonathanoesterle/djimaging.git
````

Install the package e.g. using pip as an editable package:
```bash
cd djimaging
pip install -r requirements.txt
pip install -e .
```

## Getting started
Create a datajoint config <code>*.json</code> file,
e.g. based on the template <code>djimaging/djconfig/djconf_template.json</code>.
Fill out the missing values; if you don't know how, ask someone in your group.
Never upload this personal config file to GitHub.

Inside the root folder <code>djimaging</code> (not in <code>djimaging/djimaging</code>) 
run the make user script my calling
```bash
python3 make_user.py your_username_here
```
This will create a folder <code>djimaging/user/your_username_here/notebooks</code> with some tutorial notebooks.
Note that per default everyhing in this folder will not be under version control.
Consider adding an expection for your files here, but do not upload them to the shared repository.