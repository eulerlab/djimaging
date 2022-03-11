def get_file_info(filename, datatype_loc, animal_loc, region_loc, field_loc, stimulus_loc, pharm_loc):
    """Extract information from filename"""
    file_info = filename.strip('.h5').split('_')
    datatype = file_info[datatype_loc] if len(file_info) > datatype_loc else None
    animal = file_info[animal_loc] if len(file_info) > animal_loc else None
    region = file_info[region_loc] if len(file_info) > region_loc else None
    field = file_info[field_loc] if len(file_info) > field_loc else None
    stimulus = file_info[stimulus_loc] if len(file_info) > stimulus_loc else None
    pharm = file_info[pharm_loc] if len(file_info) > pharm_loc else None

    return datatype, animal, region, field, stimulus, pharm
