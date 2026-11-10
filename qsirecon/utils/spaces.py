"""Parsing and validation for QSIRecon's ``--output-spaces`` argument.

The grammar is the one QSIPrep uses::

    token   ::= space (":" key "-" value)*
    space   ::= <TemplateFlow template name>
    key     ::= "res" | "cohort"

QSIRecon writes reconstruction outputs on the grid of the preprocessed DWI it was
given, so the DWI spaces QSIPrep accepts (``acpc``, ``anat`` and ``dwiref``) are not
requested here. An output space is a template that scalar maps are warped into, with
the input dataset's anatomical-to-template transforms. Several spaces, and several
resolutions of one space, may be requested: scalar maps are written in each.

``res-`` values come in two families:

* ``<size>mm`` -- a physical size. ``p`` is the decimal point and ``x`` separates
  axes, following https://github.com/nipreps/niworkflows/issues/997.
* a bare label -- a TemplateFlow ``res`` entity, as in fMRIPrep. The template is
  fetched at that resolution and its voxel size is kept.

Without a ``res-``, the template-space grid takes the smallest voxel dimension of the
preprocessed DWI.

``cohort-auto`` selects whichever cohort the input dataset's transforms were
estimated with, since QSIPrep is the one that chooses a cohort from the
participant's age. It is replaced with that cohort (:func:`resolve_auto_cohort`)
before the requested spaces are compared, so ``MNIInfant:cohort-2`` and
``MNIInfant:cohort-auto`` are one space for data normalized to cohort 2.

No BIDSLayout is ever constructed here and nothing needs a BIDS tree on disk, so
every function below is directly unit-testable.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

#: What --output-spaces means when it is not given.
DEFAULT_OUTPUT_SPACES = ('MNI152NLin2009cAsym',)
#: Spaces QSIPrep resamples the DWI into. QSIRecon keeps the grid of its input.
DWI_SPACES = ('acpc', 'anat', 'dwiref')
#: Templates of infant brains, which need a tighter field of view.
INFANT_TEMPLATES = ('MNIInfant', 'UNCInfant')
VALID_KEYS = ('res', 'cohort')

_SIZE = r'[0-9]+(?:p[0-9]+)?'
_MM_RE = re.compile(rf'^(?P<sizes>{_SIZE}(?:x{_SIZE}){{0,2}})mm$')


class OutputSpacesError(ValueError):
    """An ``--output-spaces`` token could not be parsed or validated."""


@dataclass(frozen=True)
class Resolution:
    """A requested output resolution.

    ``kind`` is ``'mm'`` (an explicit physical size) or ``'label'`` (a TemplateFlow
    ``res`` entity). ``label`` is always the spec exactly as the user wrote it.
    """

    kind: str
    label: str
    zooms: tuple | None = None


@dataclass(frozen=True)
class SpaceSpec:
    """One requested output space."""

    space: str
    resolution: Resolution | None = None
    cohort: str | None = None

    @property
    def fullname(self) -> str:
        """NiPreps-style name with the cohort folded in, as in ``from-``/``to-`` labels."""
        if self.cohort in (None, 'auto'):
            return self.space
        return f'{self.space}+{self.cohort}'

    @property
    def is_infant(self) -> bool:
        """Whether this is a template of infant brains."""
        return self.space in INFANT_TEMPLATES

    @property
    def templateflow_resolution(self) -> str:
        """The TemplateFlow ``res`` label to fetch the template at.

        A physical size has no TemplateFlow label, so the highest-resolution grid
        (``res-1``) is fetched and resampled.
        """
        if self.resolution is not None and self.resolution.kind == 'label':
            return self.resolution.label
        return '1'

    @property
    def res_label(self) -> str | None:
        """The value of the ``res`` entity in the names of outputs in this space.

        An output carries ``res-`` whenever a resolution was requested for its space,
        and none when the voxel size was left to follow the preprocessed DWI.
        """
        if self.resolution is None:
            return None

        return self.resolution.label

    @property
    def node_label(self) -> str:
        """This spec, spelled so that it can be part of a workflow node name."""
        return re.sub(r'[^A-Za-z0-9]+', '_', str(self))

    @classmethod
    def from_token(cls, token: str) -> 'SpaceSpec':
        """Rebuild a spec from the token :meth:`__str__` wrote, without validating it.

        Unlike :func:`parse_space_token`, this looks nothing up in TemplateFlow, so it
        is only for tokens that were already validated.
        """
        space, *rest = token.split(':')
        cohort = None
        resolution = None
        for item in rest:
            key, _, value = item.partition('-')
            if key == 'cohort':
                cohort = value
            elif key == 'res':
                mm = _MM_RE.match(value)
                if mm:
                    zooms = _parse_sizes(mm.group('sizes'))
                    resolution = Resolution(kind='mm', label=value, zooms=zooms)
                else:
                    resolution = Resolution(kind='label', label=value)

        return cls(space=space, resolution=resolution, cohort=cohort)

    def matches(self, name: str) -> bool:
        """Whether ``name``, a ``from-``/``to-`` label such as ``MNIInfant+2``, is this space."""
        template, _, cohort = name.partition('+')
        if template != self.space:
            return False

        if self.cohort == 'auto':
            return True
        return (cohort or None) == self.cohort

    def __str__(self) -> str:
        parts = [self.space]
        if self.cohort is not None:
            parts.append(f'cohort-{self.cohort}')

        if self.resolution is not None:
            parts.append(f'res-{self.resolution.label}')
        return ':'.join(parts)


def _templates() -> list:
    from templateflow.conf import TF_LAYOUT

    return TF_LAYOUT.get_templates()


def _cohorts(template: str) -> list:
    from templateflow.conf import TF_LAYOUT

    return [str(c) for c in TF_LAYOUT.get_cohorts(template=template)]


def _resolutions(template: str) -> list:
    from templateflow.conf import TF_LAYOUT

    return [str(r) for r in TF_LAYOUT.get_resolutions(template=template)]


def _parse_sizes(text: str) -> tuple:
    """Turn ``2``, ``1p5`` or ``6x6x3`` into a 3-tuple of millimeters."""
    values = []
    for chunk in text.split('x'):
        try:
            values.append(float(chunk.replace('p', '.')))
        except ValueError:
            raise OutputSpacesError(f'Could not read "{chunk}" as a voxel size.') from None

    if len(values) == 1:
        values = values * 3

    if len(values) != 3:
        raise OutputSpacesError(
            f'A res- specification needs one or three sizes, got {len(values)} in "{text}".'
        )

    if any(value <= 0 for value in values):
        raise OutputSpacesError(f'Voxel sizes must be greater than zero, got "{text}".')
    return tuple(values)


def _parse_resolution(value: str, space: str) -> Resolution:
    mm = _MM_RE.match(value)
    if mm:
        return Resolution(kind='mm', label=value, zooms=_parse_sizes(mm.group('sizes')))

    if value.startswith('native'):
        raise OutputSpacesError(
            f'res-{value} is not valid in QSIRecon. Leave the resolution off to use the '
            'smallest voxel dimension of the preprocessed DWI.'
        )

    available = _resolutions(space)
    if value not in available:
        raise OutputSpacesError(
            f'{space} has no res-{value}. Available TemplateFlow resolutions are: '
            f'{", ".join(available) or "none"}. For a custom size, add an "mm" suffix '
            f'(for example res-{value}mm).'
        )
    return Resolution(kind='label', label=value)


def _validate_cohort(space: str, cohort: str | None) -> None:
    available = _cohorts(space)
    options = ', '.join(f'cohort-{c}' for c in available)
    if cohort is None:
        if available:
            raise OutputSpacesError(
                f'{space} is not fully defined: it needs a cohort. Use cohort-auto to take '
                f'the cohort the input transforms were estimated with, or one of: {options}.'
            )
        return

    if not available:
        raise OutputSpacesError(f'{space} does not accept a cohort specification.')

    if cohort != 'auto' and cohort not in available:
        raise OutputSpacesError(
            f'{space} has no cohort-{cohort}. Available cohorts are: {options}.'
        )


def parse_space_token(token: str) -> list[SpaceSpec]:
    """Parse one ``--output-spaces`` token into one or more :class:`SpaceSpec`.

    A token carrying several ``res-`` specs expands to one spec each, following
    niworkflows#997: ``MNI152NLin2009cAsym:res-1:res-3mm`` yields two.
    """
    space, *rest = token.split(':')

    if space in DWI_SPACES:
        raise OutputSpacesError(
            f'"{space}" is not an output space in QSIRecon. Reconstruction outputs are '
            'always written on the grid of the preprocessed DWI they were estimated from, '
            'so --output-spaces only names the template space. To choose which '
            'preprocessed DWI files are reconstructed, use --bids-filter-file.'
        )

    if space not in _templates():
        raise OutputSpacesError(
            f'"{space}" is not a known output space. Use a TemplateFlow template name.'
        )

    cohort = None
    res_values = []
    for item in rest:
        key, _, value = item.partition('-')
        if not value:
            raise OutputSpacesError(f'"{item}" in "{token}" is not a key-value specification.')

        if key not in VALID_KEYS:
            raise OutputSpacesError(
                f'"{key}" in "{token}" is not a valid specifier. '
                f'QSIRecon accepts: {", ".join(VALID_KEYS)}.'
            )

        if key == 'cohort':
            if cohort is not None:
                raise OutputSpacesError(f'"{token}" specifies more than one cohort.')
            cohort = value
        else:
            res_values.append(value)

    _validate_cohort(space, cohort)

    if not res_values:
        return [SpaceSpec(space=space, resolution=None, cohort=cohort)]

    return [
        SpaceSpec(space=space, resolution=_parse_resolution(value, space), cohort=cohort)
        for value in res_values
    ]


def dedupe_specs(specs) -> list[SpaceSpec]:
    """Drop repeated specs, keeping the first occurrence of each."""
    seen = set()
    unique = []
    for spec in specs:
        key = str(spec)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def parse_output_spaces(tokens: Sequence) -> list[SpaceSpec]:
    """Parse every token into specs and de-duplicate them, keeping the order given.

    Any number of spaces, and of resolutions per space, may be requested.
    """
    return dedupe_specs(spec for token in tokens for spec in parse_space_token(token))


def resolve_auto_cohort(spec: SpaceSpec, names) -> SpaceSpec:
    """Replace ``cohort-auto`` with the cohort the input dataset was normalized to.

    Parameters
    ----------
    spec : :class:`SpaceSpec`
        A requested output space. Returned unchanged unless its cohort is ``auto``.
    names : iterable of :obj:`str`
        The templates the input dataset has anatomical-to-template transforms for,
        as ``to-`` labels such as ``MNIInfant+2``.

    Raises
    ------
    OutputSpacesError
        If the dataset has transforms to no cohort of the template, or to several.
    """
    if spec.cohort != 'auto':
        return spec

    names = sorted(set(names))
    found = [name for name in names if spec.matches(name)]
    if not found:
        raise OutputSpacesError(
            f'The output space {spec} was requested, and cohort-auto takes the cohort from '
            f'the transforms in the input dataset, but the dataset has no transform to '
            f'{spec.space}. It has transforms to: {", ".join(names) or "none"}. '
            'To reach another template through a chain of transforms, replace cohort-auto '
            'with a cohort label.'
        )

    if len(found) > 1:
        raise OutputSpacesError(
            f'The output space {spec} matches any cohort, and the input dataset has '
            f'transforms to several: {", ".join(found)}. Name one cohort in --output-spaces.'
        )

    cohort = found[0].partition('+')[2]
    return replace(spec, cohort=cohort or None)


def zooms_label(zooms) -> str:
    """Spell a voxel size the way ``res-`` values do: ``2mm``, ``1p5x1p5x3mm``."""

    def _number(value):
        return f'{float(value):g}'.replace('.', 'p')

    if len(set(zooms)) == 1:
        return f'{_number(zooms[0])}mm'
    return 'x'.join(_number(value) for value in zooms) + 'mm'


def format_zooms(zooms) -> str:
    """Describe a voxel size for people: ``2 mm isotropic``, ``1.5 x 1.5 x 3 mm``."""
    if len(set(zooms)) == 1:
        return f'{float(zooms[0]):g} mm isotropic'
    return ' x '.join(f'{float(value):g}' for value in zooms) + ' mm'
