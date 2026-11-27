# %% [markdown]
# # __PROJECT__ — __NOTEBOOK_TITLE__
#
# Notebooks of the project `__PROJECT__` (table blocks: __BLOCKS__), run them in this order:
#
__NOTEBOOK_LIST__
#
# You can use *Run all*: steps that depend on missing data or on manual work (e.g. drawing ROI masks) are skipped
# with a message instead of raising an error. Re-run the notebook once the data is available.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
os.environ["DJ_IGNORE_CONFIG_FILE"] = "true"

from pathlib import Path

import datajoint as dj

from djimaging.utils.dj_utils import activate_schema, plot_if_data
from __PACKAGE__ import settings

# %% [markdown]
# # Database
#
# The DataJoint config file and the schema name are set in `__PACKAGE__/settings.py`, so that all notebooks of this
# project use the same schema.

# %%
assert os.path.isfile(settings.CONFIG_FILE), f'Set the path to your config file in settings.py: {settings.CONFIG_FILE}'
dj.config.load(settings.CONFIG_FILE)
print("schema_name:", settings.SCHEMA_NAME)
dj.conn()

# %%
# The schema defines the tables of this project and how they are connected.
from __SCHEMA_MODULE__ import *

activate_schema(schema=schema, schema_name=settings.SCHEMA_NAME, create_schema=__CREATE_SCHEMA__, create_tables=True)
reference_root = Path(dj.config.get_store_spec('reference')['location'])
schema
