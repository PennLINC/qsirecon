"""Find a chain of transforms between the anatomical space and a template.

QSIRecon never estimates a normalization. It uses the transforms QSIPrep wrote
between the AC-PC anatomical space and each template it was asked for. When the
requested output space is not one of those, the anatomical-to-template transforms in
the input dataset can still be chained with the template-to-template transforms that
TemplateFlow distributes, as BDT does. For example, with ``from-ACPC_to-MNI152NLin2009cAsym``
in the input dataset, ``MNI152NLin6Asym`` is reached through TemplateFlow's
``tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym`` transform.

A BIDS transform named ``from-<X>_to-<Y>`` resamples an image in space X onto a grid in
space Y. TemplateFlow names the same thing ``tpl-<Y>_from-<X>``. Both are indexed here
as a directed edge ``X -> Y``.

The transforms are nonlinear, so none of them can be inverted by a flag: a hop is only
usable when the files for both directions exist. Chains are therefore searched on the
undirected graph of hops that have both files, and the chain for the opposite
direction is the same path walked backwards.

Enumerating TemplateFlow's transforms reads its manifest and downloads nothing. Only
the transforms on the chosen chain are fetched.
"""

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

#: The node of the anatomical space, whatever the input dataset calls it.
ANAT_SPACE = 'ACPC'

# tpl-<TO>[_cohort-<N>]_from-<FROM>[_mode-image]_xfm.h5
# The source label may carry a cohort (MNIInfant+2).
_TF_XFM_RE = re.compile(
    r'tpl-(?P<to>[A-Za-z0-9]+)'
    r'(?:_cohort-(?P<cohort>[A-Za-z0-9]+))?'
    r'_from-(?P<frm>[A-Za-z0-9+]+)'
    r'(?:_mode-(?P<mode>[A-Za-z0-9]+))?'
    r'_xfm\.h5$'
)


class NoTransformChainError(ValueError):
    """No chain of transforms links two spaces."""


@dataclass(frozen=True)
class Xfm:
    """One transform file, as a directed ``frm -> to`` image edge.

    ``source`` is ``'dataset'`` for a file in the input dataset and ``'templateflow'``
    for one TemplateFlow distributes, whose ``path`` may not have been fetched yet.
    """

    path: str
    frm: str
    to: str
    source: str = 'dataset'


def parse_templateflow_xfm(name) -> Xfm | None:
    """Parse the name of a TemplateFlow template-to-template transform.

    Returns None for names that are not image-mode ``.h5`` transforms.
    """
    match = _TF_XFM_RE.search(Path(name).name)
    if match is None or match.group('mode') == 'points':
        return None

    to = match.group('to')
    if match.group('cohort'):
        to = f'{to}+{match.group("cohort")}'

    return Xfm(path=str(name), frm=match.group('frm'), to=to, source='templateflow')


def _tf_templates() -> list:
    from templateflow import api

    return api.templates()


def _tf_ls(template: str) -> list:
    from templateflow import api

    return api.ls(template, suffix='xfm', extension='.h5')


def _tf_get(template: str, **query):
    from templateflow import api

    return api.get(template, **query)


def templateflow_edges() -> list[Xfm]:
    """List TemplateFlow's template-to-template transforms as edges, without downloading."""
    edges = []
    for template in _tf_templates():
        for path in _tf_ls(template):
            xfm = parse_templateflow_xfm(path)
            if xfm is not None:
                edges.append(xfm)

    return edges


def fetch_templateflow_xfm(xfm: Xfm) -> str:
    """Download a TemplateFlow transform if needed, and return its local path."""
    template, _, cohort = xfm.to.partition('+')
    query = {'from': xfm.frm, 'suffix': 'xfm', 'extension': '.h5'}
    if cohort:
        query['cohort'] = cohort

    if '_mode-' in Path(xfm.path).name:
        query['mode'] = 'image'

    found = _tf_get(template, **query)
    if not found or isinstance(found, list | tuple):
        raise FileNotFoundError(
            f'Expected one from-{xfm.frm} transform for template {xfm.to} in TemplateFlow, '
            f'but found: {found}'
        )

    return str(found)


def find_transform_chain(edges, src: str, dst: str, bidirectional: bool = True) -> list[str]:
    """Return the shortest path of spaces from ``src`` to ``dst``.

    Parameters
    ----------
    edges : iterable of :class:`Xfm`
        The available transforms.
    src, dst : :obj:`str`
        The spaces to link.
    bidirectional : :obj:`bool`, optional
        If True (the default), only hops with a transform file in each direction are
        used, so the returned path can be walked both ways.
        If False, the path only needs the files for resampling an image from ``src``
        to ``dst``.

    Returns
    -------
    path : :obj:`list` of :obj:`str`
        The spaces on the chain, from ``src`` to ``dst`` inclusive.

    Raises
    ------
    NoTransformChainError
        If no chain exists.
    """
    import networkx as nx

    directed = {(xfm.frm, xfm.to) for xfm in edges}
    # Sorted, so that ties between equally short chains are broken the same way each run
    if bidirectional:
        graph = nx.Graph()
        graph.add_edges_from(sorted((u, v) for u, v in directed if (v, u) in directed))
    else:
        graph = nx.DiGraph()
        graph.add_edges_from(sorted(directed))

    try:
        return nx.shortest_path(graph, src, dst)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise NoTransformChainError(f'No chain of transforms links {src} to {dst}.') from None


def chain_files(edges, path) -> list[Xfm]:
    """Return the transforms for resampling an image along ``path``, in ANTs order.

    ANTs applies the last-listed transform first, so the hops are returned in reverse:
    for ``ACPC -> A -> B``, ``[from-A_to-B, from-ACPC_to-A]``.

    Raises
    ------
    ValueError
        If a hop has more than one transform file.
    """
    by_hop = {}
    for xfm in edges:
        by_hop.setdefault((xfm.frm, xfm.to), []).append(xfm)

    steps = []
    for hop in pairwise(path):
        candidates = by_hop[hop]
        if len(candidates) > 1:
            files_str = '\n\t'.join(sorted(xfm.path for xfm in candidates))
            raise ValueError(
                f'More than one from-{hop[0]}_to-{hop[1]} transform found.\n'
                f'Files found:\n\t{files_str}'
            )

        steps.append(candidates[0])

    steps.reverse()
    return steps
