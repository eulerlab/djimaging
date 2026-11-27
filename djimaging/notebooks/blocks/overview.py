# %% [markdown]
# ## Important note
#
# If the schema `settings.SCHEMA_NAME` already exists, its tables must have been created from the same schema
# definition as the one imported here. If you created the schema from another project (or an earlier version of this
# one) and did not drop it, the tables differ, which can cause a variety of problems: either change `SCHEMA_NAME` in
# `settings.py` or drop the old schema first (`drop_schema` notebook).
#
# In most cases, you want exactly one schema per project to never run into this problem.

# %% [markdown]
# ## Diagram
#
# Shows all tables of your schema and how they depend on each other.

# %%
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    display(dj.Diagram(schema))
