"""Functions to generate boilerplate text for reports."""


def list_to_str(lst):
    """Convert a list to a pretty string."""
    if not lst:
        raise ValueError("Zero-length list provided.")

    lst_str = [str(item) for item in lst]
    if len(lst_str) == 1:
        return lst_str[0]
    elif len(lst_str) == 2:
        return " and ".join(lst_str)
    else:
        return f"{', '.join(lst_str[:-1])}, and {lst_str[-1]}"


def describe_atlases(atlases):
    """Build a text description of the atlases that will be used."""
    atlas_descriptions = {
        "AAL116": "",
        "AICHA384Ext": "",
        "Brainnetome246Ext": "the Brainnetome 246-parcel atlas [@Fan_2016] extended with subcortical parcels",
        "Gordon333Ext": "the Gordon 333-parcel atlas [@Gordon_2014] extended with subcortical parcels",
    }

    atlas_strings = []
    described_atlases = []
    atlases_4s = [atlas for atlas in atlases if str(atlas).startswith("4S")]
    described_atlases += atlases_4s
    if atlases_4s:
        parcels = [int(str(atlas[2:-7])) for atlas in atlases_4s]
        s = (
            "the Schaefer Supplemented with Subcortical Structures (4S) atlas "
            "[@Schaefer_2017;@pauli2018high;@king2019functional;@najdenovska2018vivo;"
            "@glasser2013minimal] "
            f"at {len(atlases_4s)} different resolutions ({list_to_str(parcels)} parcels)"
        )
        atlas_strings.append(s)

    for k, v in atlas_descriptions.items():
        if k in atlases:
            atlas_strings.append(v)
            described_atlases.append(k)

    undescribed_atlases = [atlas for atlas in atlases if atlas not in described_atlases]
    for atlas in undescribed_atlases:
        atlas_strings.append(f"the {atlas} atlas")

    return list_to_str(atlas_strings)
