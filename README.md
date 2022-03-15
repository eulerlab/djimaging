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

Checkout the notebooks in <code>djimaging/notebooks</code> 
and follow the instructions to create your first test database.