"""QSIRecon custom docstring processor for autodoc.

This extension mirrors Nipype's apidoc hookup but allows local customization.
Place modifications by subclassing the imported Nipype docstring parsers.
"""

from packaging.version import Version

import sphinx
from sphinx.ext.napoleon import _patch_python_domain

# Detect whether sphinx-design is available for grid/card/badge directives
try:  # pragma: no cover - environment-dependent
    import sphinx_design  # noqa: F401

    HAVE_SPHINX_DESIGN = True
except Exception:  # pragma: no cover - environment-dependent
    HAVE_SPHINX_DESIGN = False

# Reuse Nipype's internals so behavior stays consistent unless overridden
from nipype.sphinxext.apidoc import __version__ as NIPYPE_VERSION
from nipype.sphinxext.apidoc import _skip_member as nipype_skip_member
from nipype.sphinxext.apidoc.docstring import (  # noqa: F401
    NipypeDocstring as BaseNipypeDocstring,
    InterfaceDocstring as BaseInterfaceDocstring,
)
from nipype.sphinxext.apidoc.docstring import _parse_spec as nipype_parse_spec


class QSINipypeDocstring(BaseNipypeDocstring):
    """Hook for customizing general docstring formatting.

    Override BaseNipypeDocstring methods here to change behavior.
    Currently pass-through.
    """

    pass


class QSIInterfaceDocstring(BaseNipypeDocstring):
    """Custom Interface docstring that lists only recon-spec-accessible inputs."""

    def __init__(self, docstring, config=None, app=None, what='', name='', obj=None, options=None):
        # Initialize with the NipypeDocstring behavior (not InterfaceDocstring) to avoid
        # the default mandatory/optional sections.
        super().__init__(docstring, config, app, what, name, obj, options)

        # Prepend wrapped executable line if available (mimic Nipype behavior)
        try:
            cmd = getattr(obj, '_cmd', '')
            if cmd and isinstance(cmd, str) and cmd.strip():
                self._parsed_lines = [
                    'Wrapped executable: ``%s``.' % cmd.strip(),
                    '',
                ] + self._parsed_lines
        except Exception:
            pass

        # Append filtered interface inputs under a custom section
        try:
            if obj is not None and getattr(obj, 'input_spec', None):
                inputs = obj.input_spec()

                def _is_recon_spec_accessible(spec):
                    try:
                        if getattr(spec, 'recon_spec_accessible', None) is True:
                            return True
                        md = getattr(spec, 'metadata', None)
                        if isinstance(md, dict) and md.get('recon_spec_accessible') is True:
                            return True
                    except Exception:
                        pass
                    return False

                all_items = sorted(
                    [
                        (name, spec)
                        for name, spec in inputs.traits(transient=None).items()
                        if _is_recon_spec_accessible(spec)
                    ]
                )

                if all_items:
                    header = 'Recon Spec Options'
                    self._parsed_lines += ['', header]
                    self._parsed_lines += ['-' * len(header)]
                    # Render as styled HTML blocks (portable and compact)
                    self._parsed_lines += _format_recon_spec_blocks(inputs, all_items)

                # Always include developer information content; only show its heading
                # when Recon Spec Options are present
                self._parsed_lines += _format_developer_information(
                    obj, include_heading=bool(all_items)
                )
        except Exception:
            # Never fail the build due to docstring processing
            pass


def _process_docstring(app, what, name, obj, options, lines):
    """Custom docstring processor (replaces Nipype's handler).

    Mirrors nipype.sphinxext.apidoc's logic but uses QSI* classes
    so you can adjust behavior locally.
    """
    result_lines = lines

    try:
        from nipype.interfaces.base import BaseInterface

        if what == 'class' and isinstance(obj, type) and issubclass(obj, BaseInterface):
            result_lines[:] = QSIInterfaceDocstring(
                result_lines, app.config, app, what, name, obj, options
            ).lines()
    except Exception:
        # If nipype isn't importable or obj typing is unexpected, continue gracefully
        pass

    result_lines = QSINipypeDocstring(
        result_lines, app.config, app, what, name, obj, options
    ).lines()
    lines[:] = result_lines[:]


def setup(app):
    """Sphinx extension setup for QSIRecon's apidoc customizations."""
    # Ensure the Python domain is patched similarly to napoleon/nipype
    _patch_python_domain()

    # Autodoc must be active for this processor
    app.setup_extension('sphinx.ext.autodoc')

    # Register config values Nipype adds, so _skip_member works as expected
    # Reuse Nipype's Config._config_values for compatibility across Sphinx versions
    try:
        # Import within setup to avoid side-effects at import time
        from nipype.sphinxext.apidoc import Config as NipypeConfig

        if Version(sphinx.__version__) >= Version('8.2.1'):
            for name, default, rebuild, types in NipypeConfig._config_values:  # type: ignore[attr-defined]
                app.add_config_value(name, default, rebuild, types=types)
        else:
            for name, (default, rebuild) in NipypeConfig._config_values.items():  # type: ignore[assignment]
                app.add_config_value(name, default, rebuild)
    except Exception:
        # If anything goes wrong, continue without the extra config
        pass

    # Connect our processor with high priority to ensure it takes effect
    app.connect('autodoc-process-docstring', _process_docstring, priority=999)

    # Reuse Nipype's skip-member logic for consistency
    app.connect('autodoc-skip-member', nipype_skip_member)

    return {'version': NIPYPE_VERSION, 'parallel_read_safe': True}


def _format_recon_spec_item(inputs, name, spec):
    """Render a spec as a bullet list item (avoids two-column field lists)."""
    lines = []

    # Build description lines similar to Nipype's _parse_spec
    desc_lines = []
    try:
        if getattr(spec, 'desc', None):
            desc = ''.join([spec.desc[0].capitalize(), spec.desc[1:]])
            if not desc.endswith('.') and not desc.endswith('\n'):
                desc = '%s.' % desc
            desc_lines += desc.splitlines()
    except Exception:
        pass

    try:
        argstr = getattr(spec, 'argstr', None)
        if argstr and str(argstr).strip():
            pos = getattr(spec, 'position', None)
            if pos is None:
                desc_lines += [
                    'Maps to a command-line argument: :code:`{arg}`.'.format(
                        arg=str(argstr).strip()
                    )
                ]
            else:
                desc_lines += [
                    'Maps to a command-line argument: :code:`{arg}` (position: {pos}).'.format(
                        arg=str(argstr).strip(), pos=pos
                    )
                ]
    except Exception:
        pass

    try:
        xor = getattr(spec, 'xor', None)
        if xor:
            desc_lines += [
                'Mutually **exclusive** with inputs: %s.' % ', '.join(['``%s``' % x for x in xor])
            ]
    except Exception:
        pass

    try:
        requires = getattr(spec, 'requires', None)
        if requires:
            desc_lines += [
                '**Requires** inputs: %s.' % ', '.join(['``%s``' % x for x in requires])
            ]
    except Exception:
        pass

    try:
        if getattr(spec, 'usedefault', False):
            default = spec.default_value()[1]
            if isinstance(default, (bytes, str)) and not default:
                default = '""'
            desc_lines += ['(Default value: ``%s``)' % str(default)]
    except Exception:
        pass

    # First line: bullet with name and type info
    try:
        type_info = spec.full_info(inputs, name, None)
    except Exception:
        type_info = ''

    if type_info:
        lines.append(f'- {name} : {type_info}')
    else:
        lines.append(f'- {name}')

    # Indent description under the bullet point
    for dl in desc_lines:
        lines.append(f'  {dl}')

    return lines


def _format_recon_spec_card(inputs, name, spec):
    """Render a spec as a sphinx-design card within a grid."""
    lines = []

    # Type info
    try:
        type_info = spec.full_info(inputs, name, None)
    except Exception:
        type_info = ''

    title = f'{name}' + (f' ({type_info})' if type_info else '')
    lines.append('')
    lines.append(f'   .. grid-item-card:: {title}')
    lines.append('      :link: none')
    lines.append('')

    # Description paragraph
    try:
        if getattr(spec, 'desc', None):
            desc = ''.join([spec.desc[0].capitalize(), spec.desc[1:]])
            if not desc.endswith('.') and not desc.endswith('\n'):
                desc = f'{desc}.'
            for dl in desc.splitlines():
                lines.append(f'      {dl}')
            lines.append('')
    except Exception:
        pass

    # Badges row: CLI and Default (inline roles are robust within docstrings)
    try:
        argstr = getattr(spec, 'argstr', None)
        if argstr and str(argstr).strip():
            lines += [f'      :bdg-secondary:`CLI` {str(argstr).strip()}']
    except Exception:
        pass

    try:
        if getattr(spec, 'usedefault', False):
            default = spec.default_value()[1]
            if isinstance(default, (bytes, str)) and not default:
                default = '""'
            lines += [f'      :bdg-info:`Default` {str(default)}']
    except Exception:
        pass

    return lines


def _format_developer_information(obj, include_heading: bool = True):
    """Render the original mandatory/optional inputs and outputs for developers.

    Parameters
    ----------
    include_heading : bool
        If True, include the \"Developer Information\" heading. If False, only the
        content blocks are emitted without a section title.
    """
    lines = ['']
    try:
        if include_heading:
            header = 'Developer Information'
            lines += [header]
            lines += ['-' * len(header)]

        # Inputs
        if getattr(obj, 'input_spec', None):
            inputs = obj.input_spec()
            mandatory_items = sorted(inputs.traits(mandatory=True).items())
            if mandatory_items:
                lines += ['', 'Mandatory Inputs']
                lines += ['-' * len('Mandatory Inputs')]
                for name, spec in mandatory_items:
                    lines += nipype_parse_spec(inputs, name, spec)

            mandatory_keys = {item[0] for item in mandatory_items}
            optional_items = sorted(
                [
                    (name, val)
                    for name, val in inputs.traits(transient=None).items()
                    if name not in mandatory_keys
                ]
            )
            if optional_items:
                lines += ['', 'Optional Inputs']
                lines += ['-' * len('Optional Inputs')]
                for name, spec in optional_items:
                    lines += nipype_parse_spec(inputs, name, spec)

        # Outputs
        if getattr(obj, 'output_spec', None):
            outputs = sorted(obj.output_spec().traits(transient=None).items())
            if outputs:
                lines += ['', 'Outputs']
                lines += ['-' * len('Outputs')]
                # Nipype uses inputs in parse_spec for type context; do the same
                inputs_ctx = obj.input_spec() if getattr(obj, 'input_spec', None) else None
                for name, spec in outputs:
                    lines += nipype_parse_spec(inputs_ctx, name, spec)
    except Exception:
        pass

    return lines


def _format_recon_spec_table(inputs, items):
    """Render recon-spec-accessible traits as a Sphinx list-table."""
    lines = []
    lines += [
        '',
        '.. list-table::',
        '   :widths: 18 14 24 14 30',
        '   :header-rows: 1',
        '',
        '   * - Option',
        '     - Type',
        '     - CLI',
        '     - Default',
        '     - Description',
    ]

    for name, spec in items:
        # Type
        try:
            type_info = spec.full_info(inputs, name, None) or ''
        except Exception:
            type_info = ''

        # CLI
        try:
            argstr = getattr(spec, 'argstr', None)
            cli_text = f':code:`{str(argstr).strip()}`' if argstr and str(argstr).strip() else ''
        except Exception:
            cli_text = ''

        # Default
        try:
            if getattr(spec, 'usedefault', False):
                default = spec.default_value()[1]
                if isinstance(default, (bytes, str)) and not default:
                    default = '""'
                default_text = f'``{default}``'
            else:
                default_text = ''
        except Exception:
            default_text = ''

        # Description (single paragraph)
        try:
            if getattr(spec, 'desc', None):
                desc = ''.join([spec.desc[0].capitalize(), spec.desc[1:]])
                if not desc.endswith('.') and not desc.endswith('\n'):
                    desc = f'{desc}.'
                desc_lines = desc.splitlines()
            else:
                desc_lines = ['']
        except Exception:
            desc_lines = ['']

        # Emit row
        lines += [
            '   * - {}'.format(name),
            '     - {}'.format(type_info),
            '     - {}'.format(cli_text),
            '     - {}'.format(default_text),
            '     - {}'.format(desc_lines[0]),
        ]
        # Any additional description lines
        for extra in desc_lines[1:]:
            lines.append('       {}'.format(extra))

    return lines


def _format_recon_spec_blocks(inputs, items):
    """Render recon-spec-accessible traits as custom HTML blocks with CSS classes."""
    lines = [
        '',
        '.. raw:: html',
        '',
        '   <div class="recon-spec-list">',
    ]

    for name, spec in items:
        # Type
        try:
            type_info = spec.full_info(inputs, name, None) or ''
        except Exception:
            type_info = ''

        # CLI
        try:
            argstr = getattr(spec, 'argstr', None)
            cli_text = str(argstr).strip() if argstr and str(argstr).strip() else ''
        except Exception:
            cli_text = ''

        # Default
        try:
            if getattr(spec, 'usedefault', False):
                default = spec.default_value()[1]
                if isinstance(default, (bytes, str)) and not default:
                    default = '""'
                default_text = str(default)
            else:
                default_text = ''
        except Exception:
            default_text = ''

        # Description (single paragraph)
        try:
            if getattr(spec, 'desc', None):
                desc = ''.join([spec.desc[0].capitalize(), spec.desc[1:]])
                if not desc.endswith('.') and not desc.endswith('\n'):
                    desc = f'{desc}.'
            else:
                desc = ''
        except Exception:
            desc = ''

        # Emit HTML block for the item
        name_html = f'     <div class="recon-spec-name"><code>{name}</code>'
        if type_info:
            name_html += f' <span class="recon-spec-type">({type_info})</span>'
        name_html += '</div>'

        lines += [
            '   <div class="recon-spec-item">',
            name_html,
            f'     <div class="recon-spec-desc">{desc}</div>',
        ]
        if cli_text:
            lines.append(
                '     <div class="recon-spec-cli"><span class="recon-label">CLI</span>'
                + f'<code>{cli_text}</code></div>'
            )
        if default_text:
            lines.append(
                '     <div class="recon-spec-default"><span class="recon-label">Default</span>'
                + f'<code>{default_text}</code></div>'
            )
        lines.append('   </div>')

    lines += [
        '   </div>',
        '',
    ]

    return lines
