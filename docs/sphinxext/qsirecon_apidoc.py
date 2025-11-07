"""QSIRecon custom docstring processor for autodoc.

This extension mirrors Nipype's apidoc hookup but allows local customization.
Place modifications by subclassing the imported Nipype docstring parsers.
"""

from packaging.version import Version

import sphinx
from sphinx.ext.napoleon import _patch_python_domain

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

    def __init__(
        self, docstring, config=None, app=None, what="", name="", obj=None, options=None
    ):
        # Initialize with the NipypeDocstring behavior (not InterfaceDocstring) to avoid
        # the default mandatory/optional sections.
        super().__init__(docstring, config, app, what, name, obj, options)

        # Prepend wrapped executable line if available (mimic Nipype behavior)
        try:
            cmd = getattr(obj, "_cmd", "")
            if cmd and isinstance(cmd, str) and cmd.strip():
                self._parsed_lines = [
                    "Wrapped executable: ``%s``." % cmd.strip(),
                    "",
                ] + self._parsed_lines
        except Exception:
            pass

        # Append filtered interface inputs under a custom section
        try:
            if obj is not None and getattr(obj, "input_spec", None):
                inputs = obj.input_spec()

                def _is_recon_spec_accessible(spec):
                    try:
                        if getattr(spec, "recon_spec_accessible", None) is True:
                            return True
                        md = getattr(spec, "metadata", None)
                        if isinstance(md, dict) and md.get("recon_spec_accessible") is True:
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
                    header = "Recon Spec Options"
                    self._parsed_lines += ["", header]
                    self._parsed_lines += ["-" * len(header)]
                    for name, spec in all_items:
                        self._parsed_lines += nipype_parse_spec(inputs, name, spec)
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

        if what == "class" and isinstance(obj, type) and issubclass(obj, BaseInterface):
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
    app.setup_extension("sphinx.ext.autodoc")

    # Register config values Nipype adds, so _skip_member works as expected
    # Reuse Nipype's Config._config_values for compatibility across Sphinx versions
    try:
        # Import within setup to avoid side-effects at import time
        from nipype.sphinxext.apidoc import Config as NipypeConfig

        if Version(sphinx.__version__) >= Version("8.2.1"):
            for name, default, rebuild, types in NipypeConfig._config_values:  # type: ignore[attr-defined]
                app.add_config_value(name, default, rebuild, types=types)
        else:
            for name, (default, rebuild) in NipypeConfig._config_values.items():  # type: ignore[assignment]
                app.add_config_value(name, default, rebuild)
    except Exception:
        # If anything goes wrong, continue without the extra config
        pass

    # Connect our processor with high priority to ensure it takes effect
    app.connect("autodoc-process-docstring", _process_docstring, priority=999)

    # Reuse Nipype's skip-member logic for consistency
    app.connect("autodoc-skip-member", nipype_skip_member)

    return {"version": NIPYPE_VERSION, "parallel_read_safe": True}


