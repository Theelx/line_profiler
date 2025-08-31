"""
This module defines the |lprun| and |lprun_all| IPython magic functions.

If you are using IPython, there is an implementation of an |lprun| magic
command which will let you specify functions to profile and a statement
to execute. It will also add its
:py:class:`~.LineProfiler` instance into the |builtins|, but typically,
you would not use it like that.

You can also use |lprun_all|, which profiles the whole cell you're
executing automagically, without needing to specify lines/functions
yourself. It's meant for easier use for beginners.

For IPython 0.11+, you can install it by editing the IPython configuration file
``~/.ipython/profile_default/ipython_config.py`` to add the ``'line_profiler'``
item to the extensions list::

    c.TerminalIPythonApp.extensions = [
        'line_profiler',
    ]

Or explicitly call::

    %load_ext line_profiler

To get usage help for |lprun| and |lprun_all|, use the standard IPython
help mechanism::

    In [1]: %lprun?

.. |lprun| replace:: :py:data:`%lprun <LineProfilerMagics.lprun>`
.. |lprun_all| replace:: :py:data:`%%lprun_all <LineProfilerMagics.lprun_all>`
.. |builtins| replace:: :py:mod:`__builtins__ <builtins>`
"""

import ast
import builtins
import functools
import inspect
import os
import tempfile
import textwrap
import time
import types
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from io import StringIO
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:  # pragma: no cover
    from typing import (Callable, ParamSpec,  # noqa: F401
                        Any, ClassVar, TypeVar)

    PS = ParamSpec('PS')
    PD = TypeVar('PD', bound='_PatchDict')
    DefNode = TypeVar('DefNode', ast.FunctionDef, ast.AsyncFunctionDef)

from IPython.core.getipython import get_ipython
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
from IPython.core.page import page
from IPython.utils.ipstruct import Struct
from IPython.core.error import UsageError

from line_profiler import line_profiler, LineProfiler, LineStats
from line_profiler.autoprofile.ast_tree_profiler import AstTreeProfiler


__all__ = ('LineProfilerMagics',)

_LPRUN_ALL_CODE_OBJ_NAME = '<lprof_cell>'


@dataclass
class _ParseParamResult:
    """ Class for holding parsed info relevant to the behaviors of both
    the ``%lprun`` and ``%%lprun_all`` magics.

    Attributes:
        ``.opts``
            :py:class:`IPython.utils.ipstruct.Struct` object.
        ``.arg_str``
            :py:class:`str` of unparsed argument(s).
        ``.dump_raw_dest``
            (Descriptor) :py:class:`pathlib.Path` to write the raw
            (pickled) profiling results to, or :py:data:`None` if not to
            be written.
        ``.dump_text_dest``
            (Descriptor) :py:class:`pathlib.Path` to write the
            plain-text profiling results to, or :py:data:`None` if not
            to be written.
        ``.output_unit``
            (Descriptor) Unit to normalize the output of
            :py:meth:`line_profiler.LineProfiler.print_stats` to, or
            :py:data:`None` if not specified.
        ``.strip_zero``
            (Descriptor) Whether to call
            :py:meth:`line_profiler.LineProfiler.print_stats` with
            ``stripzeros=True``.
        ``.return_profiler``
            (Descriptor) Whether the
            :py:class:`line_profiler.LineProfiler` instance is to be
            returned.
    """
    opts: Struct
    arg_str: str

    def __getattr__(self, attr):  # type: (str) -> Any
        """ Defers to :py:attr:`_ParseParamResult.opts`."""
        return getattr(self.opts, attr)

    @functools.cached_property
    def dump_raw_dest(self):  # type: () -> Path | None
        path = self.opts.D[0]
        if path:
            return Path(path)
        return None

    @functools.cached_property
    def dump_text_dest(self):  # type: () -> Path | None
        path = self.opts.T[0]
        if path:
            return Path(path)
        return None

    @functools.cached_property
    def output_unit(self):  # type: () -> float | None
        if self.opts.u is None:
            return None
        try:
            return float(self.opts.u[0])
        except Exception:
            raise TypeError("Timer unit setting must be a float.")

    @functools.cached_property
    def strip_zero(self):  # type: () -> bool
        return "z" in self.opts

    @functools.cached_property
    def return_profiler(self):  # type: () -> bool
        return "r" in self.opts


@dataclass
class _RunAndProfileResult:
    """ Class for holding the results of both the ``%lprun`` and
    ``%%lprun_all`` magics.
    """
    stats: LineStats
    parse_result: _ParseParamResult
    message: Union[str, None] = None
    time_elapsed: Union[float, None] = None
    tempfile: Union[str, 'os.PathLike[str]', None] = None

    def __post_init__(self):
        if self.tempfile is not None:
            self.tempfile = Path(self.tempfile)
        self.output  # Fetch value

    def _make_show_func_wrapper(self, show_func):
        """
        Create a replacement for
        :py:func:`line_profiler.line_profiler.show_func` to be
        monkey-patched in, so that when showing the results of the
        entire cell the lines are not truncated at the end of the first
        code block.
        """
        tmp = self.tempfile
        if tmp is None:
            return show_func
        assert isinstance(tmp, Path)

        @functools.wraps(show_func)
        def show_func_wrapper(
                filename, start_lineno, func_name, *args, **kwargs):
            call = functools.partial(show_func,
                                     filename, start_lineno, func_name,
                                     *args, **kwargs)
            show_entire_module = (start_lineno == 1
                                  and func_name == _LPRUN_ALL_CODE_OBJ_NAME
                                  and tmp is not None
                                  and tmp.samefile(filename))
            if not show_entire_module:
                return call()
            with _PatchDict.from_module(
                    line_profiler, get_code_block=get_code_block_wrapper):
                return call()

        def get_code_block_wrapper(filename, lineno):
            """ Return the entire content of :py:attr:`~.tempfile`."""
            with tmp.open(mode='r') as fobj:
                return fobj.read().splitlines(keepends=True)

        return show_func_wrapper

    @functools.cached_property
    def output(self):  # type: () -> str
        with ExitStack() as stack:
            cap = stack.enter_context(StringIO())  # Trap text output
            patch_show_func = _PatchDict.from_module(
                line_profiler,
                show_func=self._make_show_func_wrapper(line_profiler.show_func))
            stack.enter_context(patch_show_func)
            self.stats.print(cap,
                             output_unit=self.parse_result.output_unit,
                             stripzeros=self.parse_result.strip_zero)
            return cap.getvalue().rstrip()


class _PatchProfilerIntoBuiltins:
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:IPython)
        >>> import builtins
        >>> from line_profiler import LineProfiler
        >>>
        >>>
        >>> prof = LineProfiler()
        >>> with _PatchProfilerIntoBuiltins(prof):
        ...     assert builtins.profile is prof
        ...
        >>> print(builtins.profile)
        Traceback (most recent call last):
          ...
        AttributeError: ...

    Note:
        This class doesn't itself need :py:mod:`IPython`, but it
        resides in a module that does. To reduce complications, we just
        skip this doctest if :py:mod:`IPython` (and hence this module)
        can't be imported.
    """
    def __init__(self, prof=None):
        # type: (LineProfiler | None) -> None
        if prof is None:
            prof = LineProfiler()
        self.prof = prof
        self._ctx = _PatchDict.from_module(builtins, profile=self.prof)

    def __enter__(self):  # type: () -> LineProfiler
        self._ctx.__enter__()
        return self.prof

    def __exit__(self, *a, **k):
        return self._ctx.__exit__(*a, **k)


class _PatchDict:
    def __init__(self, namespace, /, **kwargs):
        # type: (dict[str, Any], Any) -> None
        self.namespace = namespace
        self.replacements = kwargs
        self._stack = []  # type: list[dict[str, Any]]
        self._absent = object()

    def __enter__(self):  # type: (PD) -> PD
        self._push()
        return self

    def __exit__(self, *_, **__):
        self._pop()

    def _push(self):
        entry = {}
        namespace = self.namespace
        absent = self._absent
        for key, value in self.replacements.items():
            entry[key] = namespace.pop(key, absent)
            namespace[key] = value
        self._stack.append(entry)

    def _pop(self):
        namespace = self.namespace
        absent = self._absent
        for key, value in self._stack.pop().items():
            if value is absent:
                namespace.pop(key, None)
            else:
                namespace[key] = value

    @classmethod
    def from_module(cls, module, /, **kwargs):
        # type: (type[PD], types.ModuleType, Any) -> PD
        return cls(vars(module), **kwargs)


@dataclass
class _ParseParamResult:
    """ Class for holding parsed info relevant to the behaviors of both
    the ``%lprun`` and ``%%lprun_all`` magics.

    Attributes:
        ``.opts``
            :py:class:`IPython.utils.ipstruct.Struct` object.
        ``.arg_str``
            :py:class:`str` of unparsed argument(s).
        ``.dump_raw_dest``
            (Descriptor) :py:class:`pathlib.Path` to write the raw
            (pickled) profiling results to, or :py:data:`None` if not to
            be written.
        ``.dump_text_dest``
            (Descriptor) :py:class:`pathlib.Path` to write the
            plain-text profiling results to, or :py:data:`None` if not
            to be written.
        ``.output_unit``
            (Descriptor) Unit to normalize the output of
            :py:meth:`line_profiler.LineProfiler.print_stats` to, or
            :py:data:`None` if not specified.
        ``.strip_zero``
            (Descriptor) Whether to call
            :py:meth:`line_profiler.LineProfiler.print_stats` with
            ``stripzeros=True``.
        ``.return_profiler``
            (Descriptor) Whether the
            :py:class:`line_profiler.LineProfiler` instance is to be
            returned.
    """
    opts: Struct
    arg_str: str

    def __getattr__(self, attr):  # type: (str) -> Any
        """ Defers to :py:attr:`_ParseParamResult.opts`."""
        return getattr(self.opts, attr)

    @functools.cached_property
    def dump_raw_dest(self):  # type: () -> Path | None
        path = self.opts.D[0]
        if path:
            return Path(path)
        return None

    @functools.cached_property
    def dump_text_dest(self):  # type: () -> Path | None
        path = self.opts.T[0]
        if path:
            return Path(path)
        return None

    @functools.cached_property
    def output_unit(self):  # type: () -> float | None
        if self.opts.u is None:
            return None
        try:
            return float(self.opts.u[0])
        except Exception:
            raise TypeError("Timer unit setting must be a float.")

    @functools.cached_property
    def strip_zero(self):  # type: () -> bool
        return "z" in self.opts

    @functools.cached_property
    def return_profiler(self):  # type: () -> bool
        return "r" in self.opts


@dataclass
class _RunAndProfileResult:
    """ Class for holding the results of both the ``%lprun`` and
    ``%%lprun_all`` magics.
    """
    stats: LineStats
    parse_result: _ParseParamResult
    message: Union[str, None] = None
    time_elapsed: Union[float, None] = None
    tempfile: Union[str, 'os.PathLike[str]', None] = None

    def __post_init__(self):
        if self.tempfile is not None:
            self.tempfile = Path(self.tempfile)
        self.output  # Fetch value

    def _make_show_func_wrapper(self, show_func):
        """
        Create a replacement for
        :py:func:`line_profiler.line_profiler.show_func` to be
        monkey-patched in, so that when showing the results of the
        entire cell the lines are not truncated at the end of the first
        code block.
        """
        tmp = self.tempfile
        if tmp is None:
            return show_func
        assert isinstance(tmp, Path)

        @functools.wraps(show_func)
        def show_func_wrapper(
                filename, start_lineno, func_name, *args, **kwargs):
            call = functools.partial(show_func,
                                     filename, start_lineno, func_name,
                                     *args, **kwargs)
            show_entire_module = (start_lineno == 1
                                  and func_name == _LPRUN_ALL_CODE_OBJ_NAME
                                  and tmp is not None
                                  and tmp.samefile(filename))
            if not show_entire_module:
                return call()
            with _PatchDict.from_module(
                    line_profiler, get_code_block=get_code_block_wrapper):
                return call()

        def get_code_block_wrapper(filename, lineno):
            """ Return the entire content of :py:attr:`~.tempfile`."""
            with tmp.open(mode='r') as fobj:
                return fobj.read().splitlines(keepends=True)

        return show_func_wrapper

    @functools.cached_property
    def output(self):  # type: () -> str
        with ExitStack() as stack:
            cap = stack.enter_context(StringIO())  # Trap text output
            patch_show_func = _PatchDict.from_module(
                line_profiler,
                show_func=self._make_show_func_wrapper(line_profiler.show_func))
            stack.enter_context(patch_show_func)
            self.stats.print(cap,
                             output_unit=self.parse_result.output_unit,
                             stripzeros=self.parse_result.strip_zero)
            return cap.getvalue().rstrip()


class _PatchProfilerIntoBuiltins:
    """
    Example:
        >>> import builtins
        >>> from line_profiler import LineProfiler
        >>>
        >>>
        >>> prof = LineProfiler()
        >>> with _PatchProfilerIntoBuiltins(prof):
        ...     assert builtins.profile is prof
        ...
        >>> print(builtins.profile)
        Traceback (most recent call last):
          ...
        AttributeError: ...
    """
    def __init__(self, prof=None):
        # type: (LineProfiler | None) -> None
        if prof is None:
            prof = LineProfiler()
        self.prof = prof
        self._ctx = _PatchDict.from_module(builtins, profile=self.prof)

    def __enter__(self):  # type: () -> LineProfiler
        self._ctx.__enter__()
        return self.prof

    def __exit__(self, *a, **k):
        return self._ctx.__exit__(*a, **k)


class _PatchDict:
    def __init__(self, namespace, /, **kwargs):
        # type: (dict[str, Any], Any) -> None
        self.namespace = namespace
        self.replacements = kwargs
        self._stack = []  # type: list[dict[str, Any]]
        self._absent = object()

    def __enter__(self):  # type: (PD) -> PD
        self._push()
        return self

    def __exit__(self, *_, **__):
        self._pop()

    def _push(self):
        entry = {}
        namespace = self.namespace
        absent = self._absent
        for key, value in self.replacements.items():
            entry[key] = namespace.pop(key, absent)
            namespace[key] = value
        self._stack.append(entry)

    def _pop(self):
        namespace = self.namespace
        absent = self._absent
        for key, value in self._stack.pop().items():
            if value is absent:
                namespace.pop(key, None)
            else:
                namespace[key] = value

    @classmethod
    def from_module(cls, module, /, **kwargs):
        # type: (type[PD], types.ModuleType, Any) -> PD
        return cls(vars(module), **kwargs)


class ParamCube:
    """
    Order-agnostic, multi-dimensional associative cube for parameter sweeps.

    - _dims: list[str] - dimension names (logical order; can be pivoted)
    - _data: dict[tuple(values aligned with _dims) -> scalar]
    - _values: dict[str, list[Any]] - allowed values per dimension (keeps declared order)
    """

    __slots__ = ("_dims", "_data", "_values")

    def __init__(self, dims, data, values_per_dim):
        self._dims = list(dims)
        self._data = data  # mapping from tuple aligned with self._dims -> scalar
        self._values = {k: list(v) for k, v in values_per_dim.items()}

    def _with(self, dims=None, data=None, values_per_dim=None):
        return ParamCube(
            dims if dims is not None else self._dims,
            data if data is not None else self._data,
            values_per_dim if values_per_dim is not None else self._values,
        )

    def _reorder(self, first_dim):
        """Return a view with `first_dim` moved to front (no data copy)."""
        if first_dim not in self._dims:
            raise KeyError(f"Unknown dimension: {first_dim}")
        if self._dims and self._dims[0] == first_dim:
            return self
        new_dims = [first_dim] + [d for d in self._dims if d != first_dim]
        idx_map = [self._dims.index(d) for d in new_dims]
        new_data = {}
        for tup, val in self._data.items():
            new_tup = tuple(tup[i] for i in idx_map)
            new_data[new_tup] = val
        return ParamCube(new_dims, new_data, self._values)

    def _dict_from_1d(self):
        """Materialize current 1D cube as {value: scalar} respecting order."""
        if len(self._dims) != 1:
            raise KeyError("Internal: not a 1D cube")
        dim = self._dims[0]
        out = {}
        for v in self._values[dim]:
            # pull scalar corresponding to (v,)
            sub = {}
            for t, scalar in self._data.items():
                if t and t[0] == v:
                    sub[t[1:]] = scalar
            out[v] = sub[()]  # scalar
        return out

    def _filter_first_dim(self, allowed_values):
        """Filter along the first dimension to allowed_values (preserving order)."""
        if not self._dims:
            raise KeyError("No dimensions left to filter")
        first = self._dims[0]
        allowed_set = set(allowed_values)
        new_values = [v for v in self._values[first] if v in allowed_set]
        new_data = {t: val for t, val in self._data.items() if t and t[0] in allowed_set}
        new_values_per_dim = dict(self._values)
        new_values_per_dim[first] = new_values
        return ParamCube(self._dims, new_data, new_values_per_dim)

    def _select_first_value(self, value):
        """
        Select a single value along the first dimension.
        Returns:
          - scalar if no dims remain
          - dict {remaining_value: scalar} if exactly 1 dim remains
          - ParamCube otherwise
        """
        if not self._dims:
            if () in self._data:
                return self._data[()]
            raise KeyError("No data")
        first = self._dims[0]
        if value not in self._values[first]:
            raise KeyError(f"Value {value!r} not in dimension {first!r}")

        new_dims = self._dims[1:]
        new_data = {}
        for t, val in self._data.items():
            if t and t[0] == value:
                new_data[t[1:]] = val

        new_values_per_dim = {d: self._values[d] for d in new_dims}

        if not new_dims:
            return new_data[()]
        elif len(new_dims) == 1:
            dim = new_dims[0]
            out = {}
            for v in new_values_per_dim[dim]:
                sub = {}
                for t, scalar in new_data.items():
                    if t and t[0] == v:
                        sub[t[1:]] = scalar
                out[v] = sub[()]
            return out
        else:
            return ParamCube(new_dims, new_data, new_values_per_dim)

    def __getitem__(self, key):
        """
        - str:
            * if it's a dimension name -> pivot to make that dimension first;
              if that leaves a 1D cube, return {value: scalar}
            * else, if it's a value in the FIRST dimension -> select that value
        - list/tuple/set -> select subset along FIRST dimension (preserve order)
        - other scalars -> select a single value along FIRST dimension,
                          returning dict if 1D or scalar if 0D
        """
        if isinstance(key, str):
            if key in self._dims:  # pivot by dimension name
                cube = self._reorder(key)
                if len(cube._dims) == 1:
                    return cube._dict_from_1d()
                return cube
            # treat as value in the first dimension (after any pivot)
            if self._dims and key in self._values[self._dims[0]]:
                return self._select_first_value(key)
            raise KeyError(f"Unknown dimension/value: {key!r}")

        if isinstance(key, (list, tuple, set)):
            return self._filter_first_dim(key)

        return self._select_first_value(key)

    # dot-indexing:
    # - if attribute matches a dimension name, pivot (and auto-dict if 1D)
    # - else, if attribute matches a VALUE in the FIRST dimension, select it
    def __getattr__(self, name):
        if name in {"_dims", "_data", "_values"}:
            raise AttributeError(name)
        if name in self._dims:
            cube = self._reorder(name)
            if len(cube._dims) == 1:
                return cube._dict_from_1d()
            return cube
        if self._dims and name in self._values[self._dims[0]]:
            return self._select_first_value(name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def dims(self):
        """Tuple of dimension names in current order."""
        return tuple(self._dims)

    def values(self, dim_name):
        """Tuple of allowed labels for a dimension (not timings)."""
        return tuple(self._values[dim_name])

    def to_dict(self):
        """Materialize as nested dicts in current dimension order."""
        def build(dims, data):
            if not dims:
                return data[()]
            first, *rest = dims
            out = {}
            for v in self._values[first]:
                sub = {t[1:]: scalar for t, scalar in data.items() if t and t[0] == v}
                if not rest:
                    out[v] = sub[()]
                else:
                    out[v] = build(rest, sub)
            return out
        return build(self._dims, self._data)

    def item(self):
        """Return the scalar when fully selected (0D)."""
        if self._dims:
            raise ValueError("Cube is not a scalar; select all dimensions first.")
        return self._data[()]

    def __repr__(self):
        shape = "×".join(str(len(self._values[d])) for d in self._dims) or "scalar"
        return f"ParamCube(dims={self._dims}, shape={shape})"


# Allowed primitive scalar types for grid values
_PRIMITIVE_TYPES = (int, float, complex, str, bytes, bool)


@magics_class
class LineProfilerMagics(Magics):
    def _parse_parameters(self, parameter_s, getopt_spec, opts_def):
        # type: (str, str, Struct) -> _ParseParamResult
        # FIXME: There is a chance that this handling will need to be
        # updated to handle single-quoted characters better (#382)
        parameter_s = parameter_s.replace('"', r"\"").replace("'", r"\"")

        opts, arg_str = self.parse_options(
            parameter_s, getopt_spec, list_all=True)
        opts.merge(opts_def)
        return _ParseParamResult(opts, arg_str)

    @staticmethod
    def _run_and_profile(prof,  # type: LineProfiler
                         parse_result,  # type: _ParseParamResult
                         tempfile,  # type: str | None
                         method,  # type: Callable[PS, Any]
                         *args,  # type: PS.args
                         **kwargs,  # type: PS.kwargs
                         ):  # type: (...) -> _RunAndProfileResult
        # Use the time module because it's easier than parsing the
        # output from `show_text()`.
        # `perf_counter()` is a monotonically increasing alternative to
        # `time()` that's intended for simple benchmarking.
        start_time = time.perf_counter()
        try:
            method(*args, **kwargs)
            message = None
        except (SystemExit, KeyboardInterrupt) as e:
            message = (f"{type(e).__name__} exception caught in "
                       "code being profiled.")

        # Capture and save total runtime
        total_time = time.perf_counter() - start_time
        return _RunAndProfileResult(
            prof.get_stats(), parse_result,
            message=message, time_elapsed=total_time, tempfile=tempfile)

    @classmethod
    def _lprun_all_get_rewritten_profiled_code(cls, tmpfile):
        # type: (str) -> types.CodeType
        """ Transform and compile the AST of the profiled code. This is
        similar to :py:meth:`.LineProfiler.runctx`,
        """
        at = AstTreeProfiler(tmpfile, [tmpfile], profile_imports=False)
        tree = at.profile()

        return compile(tree, tmpfile, "exec")

    @classmethod
    def _lprun_get_top_level_profiled_code(cls, tmpfile):
        # type: (str) -> types.CodeType
        """ Compile the profiled code."""
        with open(tmpfile, mode='r') as fobj:
            return compile(fobj.read(), tmpfile, "exec")

    @staticmethod
    def _handle_end(prof, run_result):
        # type: (LineProfiler, _RunAndProfileResult) -> LineProfiler | None
        page(run_result.output)

        dump_file = run_result.parse_result.dump_raw_dest
        if dump_file is not None:
            prof.dump_stats(dump_file)
            print(f"\n*** Profile stats pickled to file {str(dump_file)!r}.")

        text_file = run_result.parse_result.dump_text_dest
        if text_file is not None:
            with text_file.open("w", encoding="utf-8") as pfile:
                print(run_result.output, file=pfile)
            print("\n*** Profile printout saved to text file "
                  f"{str(text_file)!r}.")

        if run_result.message:
            print("\n*** " + run_result.message)

        return prof if run_result.parse_result.return_profiler else None

    @line_magic
    def lprun(self, parameter_s=""):
        """Execute a statement under the line-by-line profiler from the
        :py:mod:`line_profiler` module.

        Usage::

            %lprun [<options>] <statement>

        The given statement (which doesn't require quote marks) is run
        via the :py:class:`~.LineProfiler`. Profiling is enabled for
        the functions specified by the ``-f`` options. The statistics
        will be shown side-by-side with the code through the pager once
        the statement has completed.

        Options:

        ``-f <function>``: :py:class:`~.LineProfiler` only profiles
        functions and methods it is told to profile. This option tells
        the profiler about these functions. Multiple ``-f`` options may
        be used. The argument may be any expression that gives
        a Python function or method object. However, one must be
        careful to avoid spaces that may confuse the option parser.

        ``-m <module>``: Get all the functions/methods in a module

        One or more ``-f`` or ``-m`` options are required to get any
        useful results.

        ``-D <filename>``: dump the raw statistics out to a pickle file
        on disk. The usual extension for this is ``.lprof``. These
        statistics may be viewed later by running
        ``python -m line_profiler``.

        ``-T <filename>``: dump the text-formatted statistics with the
        code side-by-side out to a text file.

        ``-r``: return the :py:class:`~.LineProfiler` object after it
        has completed profiling.

        ``-s``: strip out all entries from the print-out that have
        zeros. This is an old, soon-to-be-deprecated alias for ``-z``.

        ``-z``: strip out all entries from the print-out that have
        zeros.

        ``-u``: specify time unit for the print-out in seconds.
        """
        opts_def = Struct(D=[""], T=[""], f=[], m=[], u=None)
        parsed = self._parse_parameters(parameter_s, "rszf:m:D:T:u:", opts_def)
        if "s" in parsed.opts:  # Handle alias
            parsed.opts["z"] = True

        assert self.shell is not None
        global_ns = self.shell.user_global_ns
        local_ns = self.shell.user_ns

        # Get the requested functions.
        funcs = []
        for name in parsed.f:
            try:
                funcs.append(eval(name, global_ns, local_ns))
            except Exception as e:
                raise UsageError(
                    f"Could not find function {name}.\n{e.__class__.__name__}: {e}"
                )

        profile = LineProfiler(*funcs)

        # Get the modules, too
        for modname in parsed.m:
            try:
                mod = __import__(modname, fromlist=[""])
                profile.add_module(mod)
            except Exception as e:
                raise UsageError(
                    f"Could not find module {modname}.\n{e.__class__.__name__}: {e}"
                )

        with _PatchProfilerIntoBuiltins(profile):
            run = self._run_and_profile(
                profile, parsed, None, profile.runctx, parsed.arg_str,
                globals=global_ns, locals=local_ns)

        return self._handle_end(profile, run)


    @magic_arguments()
    @argument('-r', '--repeats', type=int, default=1,
              help='Number of times to run the function under the profiler.')
    @argument('--grid', type=str, default=None,
              help='Name or expression of a dict[str,'
                   'list[int|float|complex|str|bytes|bool]] to sweep, '
                   'e.g. {"a":[10,100,1000]} or param_grid')
    @argument('call', nargs='*',
              help='Function call expression, e.g. my_func(1000, x=2)')
    @line_magic
    def lprun_n(self, line):
        """Execute the function under the line-by-line
        profiler from the :py:mod:`line_profiler` module.

        Usage::

            (arg1, arg2), timings = %lprun_n [<options>] MyFunction()

        The timings will be returned as an additional return value. That
        is to say, if the function normally returns arg1 and arg2, you'd
        access the timings through tuple deconstruction like in the
        above usage.

        Options:

        ``-r <repeats=1>``: change the number of times the function is
        run inside a loop, for averaging purposes.

        ``--grid <filename>``: Do a grid search on the specified
        arguments and values. The grid should be passed as a dictionary
        mapping the argument name to a list or tuple of parameters that
        it should try. An exhaustive grid search of all parameter
        combinations is conducted.
        """
        args = parse_argstring(self.lprun_n, line)
        call_str = ' '.join(args.call)
        if not call_str:
            raise ValueError("Provide a function call, e.g. %lprun_n -r 3 my_func(1000)")

        node = ast.parse(call_str, mode='eval').body
        if not isinstance(node, ast.Call):
            raise ValueError("Argument must be a function call expression like: my_func(1000)")

        user_ns = self.shell.user_ns

        def eval_expr(expr_node):
            code = compile(ast.Expression(expr_node), "<lprun_n>", "eval")
            return eval(code, user_ns, user_ns)

        func_obj = eval_expr(node.func)
        sig = inspect.signature(func_obj)

        base_pos_args = [eval_expr(a) for a in node.args]
        base_kwargs = {kw.arg: eval_expr(kw.value) for kw in node.keywords}

        def build_kwargs_with_overrides(override_dict=None):
            final = {}
            for name, p in sig.parameters.items():
                if p.default is not inspect._empty:
                    final[name] = p.default
            bound = sig.bind_partial(*base_pos_args, **base_kwargs)
            bound.apply_defaults()
            for (name, _p), val in zip(sig.parameters.items(), bound.args):
                final[name] = val
            final.update(bound.kwargs)
            if override_dict:
                for k, v in override_dict.items():
                    if k not in sig.parameters:
                        raise ValueError(f"Unknown parameter in grid: {k}")
                    final[k] = v
            return final

        def profile_kwargs_only(func, kw_args, repeats):
            lp = LineProfiler()
            wrapped = lp(func)
            res = None
            for _ in range(repeats):
                res = wrapped(**kw_args)  # kwargs-only call avoids multiple-values error
            stats = lp.get_stats()
            unit = stats.unit  # seconds per tick
            total_ticks = 0
            for (_fname, _first_lineno, func_name), line_list in stats.timings.items():
                if func_name == func.__name__:
                    total_ticks += sum(ticks for (_lineno, _hits, ticks) in line_list)
            return res, total_ticks * unit

        # Single-call mode
        if not args.grid:
            kw = build_kwargs_with_overrides()
            return profile_kwargs_only(func_obj, kw, args.repeats)

        # Grid mode: evaluate and validate grid
        try:
            try:
                param_grid = user_ns[args.grid]
            except KeyError:
                param_grid = eval(compile(ast.parse(args.grid, mode='eval'),
                                          "<lprun_n_grid>", "eval"),
                                  user_ns, user_ns)
        except Exception as e:
            raise ValueError(f"Could not evaluate --grid expression '{args.grid}': {e}")

        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError("--grid must be a non-empty dict[str, list[primitive]]")

        # Validate types and normalize to lists preserving order
        values_per_dim = OrderedDict()
        for k, v in param_grid.items():
            if not isinstance(k, str):
                raise ValueError(f"Grid key '{k}' must be a string")
            if not isinstance(v, (list, tuple)) or not v:
                raise ValueError(f"Grid values for '{k}' must be a non-empty list/tuple")
            for x in v:
                if not isinstance(x, _PRIMITIVE_TYPES):
                    raise ValueError(
                        f"Grid values for '{k}' must be one of {tuple(t.__name__ for t in _PRIMITIVE_TYPES)}; "
                        f"got {type(x).__name__}"
                    )
            values_per_dim[k] = list(v)

        dims = list(values_per_dim.keys())
        combos = list(product(*[values_per_dim[k] for k in dims]))

        # We will build dense maps keyed by tuple aligned with dims
        times_data = {}
        result_data_list = None  # one ParamCube per result component

        # Sweep
        for combo in combos:
            overrides = {k: v for k, v in zip(dims, combo)}
            kw = build_kwargs_with_overrides(overrides)
            res, secs = profile_kwargs_only(func_obj, kw, args.repeats)

            times_data[combo] = secs

            # Initialize result containers once we see the function's return
            if result_data_list is None:
                if isinstance(res, tuple):
                    m = len(res)
                    result_data_list = [dict() for _ in range(m)]
                else:
                    result_data_list = [dict()]  # single "column"

            if isinstance(res, tuple):
                for i, ri in enumerate(res):
                    result_data_list[i][combo] = ri
            else:
                result_data_list[0][combo] = res

        # Wrap into ParamCubes (order-agnostic, flexible indexing)
        times_cube = ParamCube(dims, times_data, values_per_dim)
        result_cubes = tuple(ParamCube(dims, dct, values_per_dim) for dct in result_data_list)

        return result_cubes, times_cube


    @cell_magic
    def lprun_all(self, parameter_s="", cell=""):
        """Execute the whole notebook cell under the line-by-line
        profiler from the :py:mod:`line_profiler` module.

        Usage::

            %%lprun_all [<options>]

        By default, without the ``-p`` option, it includes nested
        functions in the profiler. The statistics will be shown
        side-by-side with the code through the pager once the statement
        has completed.

        Options:

        ``-D <filename>``: dump the raw statistics out to a pickle file
        on disk. The usual extension for this is ``.lprof``. These
        statistics may be viewed later by running
        ``python -m line_profiler``.

        ``-T <filename>``: dump the text-formatted statistics with the
        code side-by-side out to a text file.

        ``-r``: return the :py:class:`~.LineProfiler` object after it
        has completed profiling.

        ``-z``: strip out all entries from the print-out that have
        zeros. This is included for consistency with the CLI.

        ``-u``: specify time unit for the print-out in seconds.

        ``-t``: store the total time taken (in seconds) to a variable
        called ``_total_time_taken`` in your notebook. This can be
        useful if you want to plot the total time taken for different
        versions of a code cell without needing to manually look at and
        type down the time taken. This can be accomplished with ``-r``,
        but that would require a decent bit of boilerplate code and some
        knowledge of the timings data structure, so this is added to be
        beginner-friendly.

        ``-p``: Profile only top-level code (ignore nested functions).
        Using this can bypass any issues with :py:mod:`ast`
        transformations.
        """
        opts_def = Struct(D=[""], T=[""], u=None)
        parsed = self._parse_parameters(parameter_s, "rzptD:T:u:", opts_def)

        ip = get_ipython()
        if not cell.strip():  # Edge case
            cell = "..."

        # Write the cell to a temporary file so `show_text()` inside
        # `print_stats()` can open it.
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        ) as tf:
            tf.write(textwrap.dedent(cell).strip('\n'))

        try:
            if "p" not in parsed.opts:  # This is the default case.
                get_code = self._lprun_all_get_rewritten_profiled_code
            else:
                get_code = self._lprun_get_top_level_profiled_code
            # Inject a fresh LineProfiler into @profile.
            with _PatchProfilerIntoBuiltins() as prof:
                code = get_code(tf.name).replace(
                    co_name=_LPRUN_ALL_CODE_OBJ_NAME)
                try:
                    code = code.replace(
                        co_qualname=_LPRUN_ALL_CODE_OBJ_NAME)
                except TypeError:  # Python < 3.11
                    pass
                # "Register" the profiled code object with the profiler
                # Notes:
                # - This uses a dummy "function" object in a hacky way,
                #   but it's OK since `add_function()` ultimately only
                #   looks at the object's `.__code__` or
                #   `.__func__.__code__`.
                # - `prof.add_function()` might have replaced the code
                #   object, so retrieve it back from the dummy function
                mock_func = types.SimpleNamespace(__code__=code)
                prof.add_function(mock_func)  # type: ignore[arg-type]
                code = mock_func.__code__
                # Notes:
                # - We don't define `ip.user_global_ns` and `ip.user_ns`
                #   at the beginning like in lprun because the ns
                #   changes after the previous compile call.
                # - The method `._run_and_profile()` fetches the
                #   `LineProfiler.print_stats()` output before the
                #   `os.unlink()` below happens, allowing for transient
                #   items to be profiled.
                with prof:
                    run = self._run_and_profile(
                        prof, parsed, tf.name, exec, code,
                        # `globals` and `locals`
                        ip.user_global_ns, ip.user_ns)
        finally:
            os.unlink(tf.name)
        if "t" in parsed.opts:
            # I know it seems redundant to include this because users
            # could just use -r to get the info, but see the docstring
            # for why -t is included anyway.
            ip.user_ns["_total_time_taken"] = run.time_elapsed

        return self._handle_end(prof, run)
