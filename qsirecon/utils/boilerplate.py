"""Functions to generate boilerplate text for reports."""

from nipype.interfaces.base import isdefined


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
    """Build documentation for a given interface.

    This is only useful for inputs that are specified when the interface is initialized.
    It will not work for inputs that are set through workflow connections.

    It will work with either the original interface object or a Nipype Node.
    It has not been tested with MapNodes or JoinNodes.
    """
    doc = []
    input_traits = sorted(interface.inputs.class_editable_traits())
    # Define order of traits to be documented based on doc_position field,
    # when available.
    # Otherwise, use alphabetical order.
    doc_position_traits = [
        trait
        for trait in input_traits
        if hasattr(interface.inputs.class_traits()[trait], "doc_position")
    ]
    doc_position_traits.sort(key=lambda x: interface.inputs.class_traits()[x].doc_position)
    non_doc_position_traits = sorted(
        [trait for trait in input_traits if trait not in doc_position_traits]
    )
    input_traits = doc_position_traits + non_doc_position_traits

    for _trait in input_traits:
        _trait_obj = interface.inputs.class_traits()[_trait]
        conditional_doc = _trait_obj.doc
        if isinstance(conditional_doc, ConditionalDoc):
            value = getattr(interface.inputs, _trait)
            doc.append(conditional_doc.get_doc(value))
        elif conditional_doc is None:
            continue
        else:
            raise ValueError(f"Conditional doc is not a ConditionalDoc: {conditional_doc}")

    doc = [d for d in doc if d]
    return " ".join(doc)


class ConditionalDoc:
    """Store conditional documentation for a Nipype Interface input.

    This is only useful for inputs that are specified when the interface is initialized.
    It will not work for inputs that are set through workflow connections.

    Parameters
    ----------
    if_true : str
        The string to use if the input's value is not False or _Undefined.
        This string may have a value variable to incorporate (e.g., "The value is {value}."),
        but it will still work if not.
    if_false : str, optional
        The string to use if the input's value is False.
        Default is an empty string.
    if_undefined : str or None, optional
        The string to use if the input's value is Undefined.
        If not specified, the string from `if_false` will be used instead.

    Returns
    -------
    str
        The formatted string from the appropriate input (if_true, if_false, or if_undefined).

    Examples
    ---------
    >>> doc = ConditionalDoc(if_true="A passing test.", if_false="A failing test.")
    >>> doc.get_doc(value='value')
    "A passing test."

    >>> doc = ConditionalDoc(if_true="Value of {value}.", if_false="A failing test.")
    >>> doc.get_doc(value=3000)
    "Value of 3000."
    """

    def __init__(self, if_true, if_false="", if_undefined=None):
        self.if_true = if_true
        self.if_false = if_false
        if if_undefined is None:
            self.if_undefined = self.if_false
        else:
            self.if_undefined = if_undefined

    def get_doc(self, value):
        if isdefined(value):
            if isinstance(value, bool):
                return (
                    self.if_true.format(value=value)
                    if value
                    else self.if_false.format(value=value)
                )
            return self.if_true.format(value=value)
        return self.if_undefined.format(value=value)
