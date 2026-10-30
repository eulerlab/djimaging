# %% [markdown]
# # Clean up
#
# If you are done with the tutorial you can delete (=drop) your schema again and create a schema with a more
# meaningful name than `ageuler_{username}_test`.
#
# To not delete data by accident when using *Run all*, you have to set `DROP_SCHEMA = True` explicitly.

# %%
DROP_SCHEMA = False

if DROP_SCHEMA:
    schema.drop()
else:
    print("Schema was not dropped. Set DROP_SCHEMA = True to drop it.")
