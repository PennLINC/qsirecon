"""Functions to generate boilerplate text for reports."""

from nipype.interfaces.base import traits, isdefined


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
        "AAL116": (
            "the Automated Anatomical Labeling (AAL) 116-parcel atlas [@tzourio2002automated]"
        ),
        "AICHA384Ext": (
            "the AICHA 384-parcel atlas [@joliot2015aicha] extended with subcortical parcels"
        ),
        "Brainnetome246Ext": (
            "the Brainnetome 246-parcel atlas [@fan2016human] extended with subcortical parcels"
        ),
        "Gordon333Ext": (
            "the Gordon 333-parcel atlas [@gordon2016generation] extended with subcortical "
            "parcels"
        ),
    }

    atlas_strings = []
    described_atlases = []
    atlases_4s = [atlas for atlas in atlases if str(atlas).startswith("4S")]
    described_atlases += atlases_4s
    if atlases_4s:
        parcels = [int(str(atlas[2:-7])) for atlas in atlases_4s]
        s = (
            "the Schaefer Supplemented with Subcortical Structures (4S) atlas "
            "[@schaefer2018local;@pauli2018high;@king2019functional;@najdenovska2018vivo;"
            "@hcppipelines] "
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


def build_documentation(interface):
    """Build documentation for a given interface."""
    doc = []
    input_traits = interface.inputs.class_editable_traits()
    for _trait in input_traits:
        _trait_obj = interface.inputs.class_traits()[_trait]

        if hasattr(_trait_obj, "doc"):
            conditional_doc = _trait_obj.doc
            value = getattr(interface.inputs, _trait)
            doc.append(conditional_doc.get_doc(value))

    return " ".join(doc)


class ConditionalDoc:
    def __init__(self, if_true, if_false= "", if_undefined=None):
        self.if_true = if_true
        self.if_false = if_false
        if if_undefined is None:
            self.if_undefined = self.if_false
        else:
            self.if_undefined = if_undefined

    def get_doc(self, value):
        if isdefined(value):
            if value:
                return self.if_true.format(value = value)
            return self.if_false.format(value = value)
        return self.if_undefined.format(value = value)

