"""
This module defines the ``%lprun`` and ``%%lprun_all`` IPython magic functions.

If you are using IPython, there is an implementation of an %lprun magic command
which will let you specify functions to profile and a statement to execute. It
will also add its LineProfiler instance into the __builtins__, but typically,
you would not use it like that.

You can also use %%lprun_all, which profiles the whole cell you're executing
automagically, without needing to specify lines/functions yourself. It's meant
for easier use for beginners.

For IPython 0.11+, you can install it by editing the IPython configuration file
``~/.ipython/profile_default/ipython_config.py`` to add the ``'line_profiler'``
item to the extensions list::

    c.TerminalIPythonApp.extensions = [
        'line_profiler',
    ]

Or explicitly call::

    %load_ext line_profiler

To get usage help for %lprun and %%lprun_all, use the standard IPython help mechanism::

    In [1]: %lprun?
"""

import ast
import builtins
import os
import tempfile
import textwrap
import time
from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union
if TYPE_CHECKING:
    from types import CodeType  # noqa: F401
    from typing import Callable, ParamSpec, ClassVar, TypeVar  # noqa: F401

    PS = ParamSpec('PS')
    DefNode = TypeVar('DefNode', ast.FunctionDef, ast.AsyncFunctionDef)

from io import StringIO

from IPython.core.getipython import get_ipython
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.page import page
from IPython.utils.ipstruct import Struct
from IPython.core.error import UsageError

from line_profiler import LineProfiler, LineStats
from line_profiler.autoprofile.ast_tree_profiler import AstTreeProfiler
from line_profiler.autoprofile.ast_profile_transformer import AstProfileTransformer


# This is used for profiling all the code within a cell with lprun_all
class SkipWrapper(AstProfileTransformer):
    """
    AST Transformer that lets the base transformer add @profile everywhere, then
    removes it from the wrapper function only. Helps resolve issues where only top-level
    code would show timings.
    """

    def __init__(self, *args, wrapper_name, **kwargs):
        # Yes, I know these look like ChatGPT-generated docstrings, but I wrote them
        # in order to follow the format from ./autoprofile/ast_profile_transformer.py
        """Initialize the transformer.

        The base AstProfileTransformer is expected to add `@profile` to functions.
        This subclass remembers the name of the generated wrapper function so we can
        strip @profile off the wrapper function later because we only want to profile
        the code inside the wrapper, not the wrapper itself.

        Args:
            wrapper_name (str): The exact name of the wrapper function whose
                decorators should be cleaned.
            *args: Positional args forwarded to the parent transformer.
            **kwargs: Keyword args forwarded to the parent transformer.
        """
        super().__init__(*args, **kwargs)
        self._wrapper_name = wrapper_name  # type: str

    def _strip_profile_from_decorators(self, node):
        # type: (DefNode) -> DefNode
        """Remove any @profile decorator from a function node.

        Handles both the bare decorator form (@profile) and the callable form
        (@profile(...)). The node is modified in-place by filtering its
        decorator_list.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef): The function node to clean.

        Returns:
            ast.AST: The same node instance, with @profile-related decorators removed.
        """

        def keep(d):
            # Drop the decorator if it is exactly profile
            if isinstance(d, ast.Name) and d.id == "profile":
                return False
            # Drop calls like @profile(...) too
            if (
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "profile"
            ):
                return False
            # Keep the rest
            return True

        # Filter decorators in-place because NodeTransformer expects us to return the node
        node.decorator_list = [d for d in node.decorator_list if keep(d)]
        return node

    def visit_FunctionDef(self, node):
        # type: (ast.FunctionDef) -> ast.FunctionDef
        """Visit a synchronous "def" function.

        We first delegate to the base transformer so it can apply its logic
        (e.g., adding @profile to functions). If the function happens to be the
        special wrapper (self._wrapper_name), we remove the `@profile` decorator
        from it so profiling reflects the code executed within the wrapper.

        Args:
            node (ast.FunctionDef): The function definition node.

        Returns:
            ast.FunctionDef: The possibly modified node.
        """
        node = super().visit_FunctionDef(node)
        if isinstance(node, ast.FunctionDef) and node.name == self._wrapper_name:
            node = self._strip_profile_from_decorators(node)
        return node

    # This isn't needed by our code because our _lprof_cell will never be async,
    # but it's included in case a user needs to make it async to work with their code
    def visit_AsyncFunctionDef(self, node):
        # type: (ast.AsyncFunctionDef) -> ast.AsyncFunctionDef
        """Visit an asynchronous "async def" function.

        Mirrors visit_FunctionDef but for async functions. After the base
        transformer adds @profile, we remove it from the wrapper function if
        the names match.

        Args:
            node (ast.AsyncFunctionDef): The async function definition node.

        Returns:
            ast.AsyncFunctionDef: The possibly modified node.
        """
        node = super().visit_AsyncFunctionDef(node)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == self._wrapper_name:
            node = self._strip_profile_from_decorators(node)
        return node


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

    def __getitem__(self, key):  # type: (str) -> Any
        """ Defers to :py:attr:`_ParseParamResult.opts`."""
        return self.opts[key]

    @cached_property
    def dump_raw_dest(self):  # type: () -> Path | None
        path = self.opts.D[0]
        if path:
            return Path(path)
        return None

    @cached_property
    def dump_text_dest(self):  # type: () -> Path | None
        path = self.opts.T[0]
        if path:
            return Path(path)
        return None

    @cached_property
    def output_unit(self):  # type: () -> float | None
        if self.opts.u is None:
            return None
        try:
            return float(self.opts.u[0])
        except Exception:
            raise TypeError("Timer unit setting must be a float.")

    @cached_property
    def strip_zero(self):  # type: () -> bool
        return "z" in self.opts

    @cached_property
    def return_profiler(self):  # type: () -> bool
        return "r" in self.opts


@dataclass
class _RunAndProfileResult:
    """ Class for holding the results of both the ``%lprun`` and
    ``%%lprun_all`` magics.
    """
    stats: LineStats
    parse_result: _ParseParamResult
    return_value: Any
    message: Union[str, None] = None
    time_elapsed: Union[float, None] = None

    def __post_init__(self):
        self.output  # Fetch value

    @cached_property
    def output(self):  # type: () -> str
        with StringIO() as capture:  # Trap text output
            self.stats.print(capture,
                             output_unit=self.parse_result.output_unit,
                             stripzeros=self.parse_result.strip_zero)
            return capture.getvalue().rstrip()


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
        self.prof = prof or LineProfiler()  # type: LineProfiler
        self._namespace = vars(builtins)  # type: dict[str, Any]
        self._state = False, None  # type: tuple[bool, Any]

    def __enter__(self):  # type: () -> LineProfiler
        try:
            self._state = True, self._namespace['profile']
        except KeyError:
            self._state = False, None
        # Add the profiler to the builtins for @profile.
        self._namespace['profile'] = self.prof
        return self.prof

    def __exit__(self, *_, **__):
        self._state, (had_profile, old_profile) = (False, None), self._state
        if had_profile:
            self._namespace['profile'] = old_profile
        else:
            self._namespace.pop('profile', None)


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
            return_value = method(*args, **kwargs)
            message = None
        except (SystemExit, KeyboardInterrupt) as e:
            message = (f"{type(e).__name__} exception caught in "
                       "code being profiled.")

        # Capture and save total runtime
        total_time = time.perf_counter() - start_time
        return _RunAndProfileResult(
            prof.get_stats(), parse_result, return_value, message, total_time)

    @classmethod
    def _lprun_all_get_rewritten_profiled_code(cls, tmpfile):
        # type: (str) -> CodeType
        # Run the AST transformer on the temp file, while skipping the
        # wrapper function.
        get_transformer = partial(SkipWrapper,
                                  wrapper_name=cls._lprof_all_fname)
        at = AstTreeProfiler(
            tmpfile,
            [tmpfile],
            profile_imports=False,
            ast_transformer_class_handler=get_transformer)  # type: ignore[arg-type]
        tree = at.profile()

        # Compile and exec that AST. This is similar to `prof.runctx`,
        # but that doesn't support executing AST.
        return compile(tree, tmpfile, "exec")

    @classmethod
    def _lprun_get_top_level_profiled_code(cls, tmpfile):
        # type: (str) -> CodeType
        # Compile and define the function from that file.
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
        line_profiler module.

        Usage:

            %lprun -f func1 -f func2 <statement>

        The given statement (which doesn't require quote marks) is run via the
        LineProfiler. Profiling is enabled for the functions specified by the -f
        options. The statistics will be shown side-by-side with the code through the
        pager once the statement has completed.

        Options:

        -f <function>: LineProfiler only profiles functions and methods it is told
        to profile.  This option tells the profiler about these functions. Multiple
        -f options may be used. The argument may be any expression that gives
        a Python function or method object. However, one must be careful to avoid
        spaces that may confuse the option parser.

        -m <module>: Get all the functions/methods in a module

        One or more -f or -m options are required to get any useful results.

        -D <filename>: dump the raw statistics out to a pickle file on disk. The
        usual extension for this is ".lprof". These statistics may be viewed later
        by running line_profiler.py as a script.

        -T <filename>: dump the text-formatted statistics with the code side-by-side
        out to a text file.

        -r: return the LineProfiler object after it has completed profiling.

        -s: strip out all entries from the print-out that have zeros.
        This is an old alias for -z.

        -z: strip out all entries from the print-out that have zeros.

        -u: specify time unit for the print-out in seconds.
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
                profile, parsed, profile.runctx, parsed.arg_str,
                globals=global_ns, locals=local_ns)

        return self._handle_end(profile, run)

    @cell_magic
    def lprun_all(self, parameter_s="", cell=""):
        """Execute the whole notebook cell under the line-by-line profiler from the
        line_profiler module.

        Usage:

            %%lprun_all <options>

        By default, without the -p option, it includes nested functions in the profiler.
        The statistics will be shown side-by-side with the code through the pager
        once the statement has completed.

        Options:

        -D <filename>: dump the raw statistics out to a pickle file on disk. The
        usual extension for this is ".lprof". These statistics may be viewed later
        by running line_profiler.py as a script.

        -T <filename>: dump the text-formatted statistics with the code side-by-side
        out to a text file.

        -r: return the LineProfiler object after it has completed profiling.

        -z: strip out all entries from the print-out that have zeros. Note: this is -s in
        lprun, however we use -z here for consistency with the CLI.

        -u: specify time unit for the print-out in seconds.

        -t: store the total time taken (in seconds) to a variable called
        `_total_time_taken` in your notebook. This can be useful if you want
        to plot the total time taken for different versions of a code cell without
        needing to manually look at and type down the time taken. This can be accomplished
        with -r, but that would require a decent bit of boilerplate code and some knowledge
        of the timings data structure, so this is added to be beginner-friendly.

        -p: Profile only top-level code (ignore nested functions). Using this can bypass
        any issues with ast transformation.
        """
        opts_def = Struct(D=[""], T=[""], u=None)
        parsed = self._parse_parameters(parameter_s, "rzptD:T:u:", opts_def)

        ip = get_ipython()
        # We have to encase the cell being profiled in an outer function
        # if we want this to work.
        if not cell.strip():  # Edge case
            cell = '...'
        indented = textwrap.indent(cell, "    ")
        fsrc = f"def {self._lprof_all_fname}():\n{indented}"

        # Write the cell to a temporary file so `show_text()` inside
        # `print_stats()` can open it.
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        ) as tf:
            tf.write(fsrc)

        try:
            if "p" not in parsed.opts:  # This is the default case.
                get_code = self._lprun_all_get_rewritten_profiled_code
            else:
                get_code = self._lprun_get_top_level_profiled_code
            # Inject a fresh LineProfiler into @profile.
            with _PatchProfilerIntoBuiltins() as prof:
                # We don't define `ip.user_global_ns` and `ip.user_ns`
                # at the beginning like in lprun because the ns changes
                # after the previous compile call.
                exec(get_code(tf.name), ip.user_global_ns, ip.user_ns)
                try:
                    # Grab and call the wrapper so it actually runs
                    # under @profile.
                    func = ip.user_ns[self._lprof_all_fname]
                except KeyError:
                    raise RuntimeError(
                        f"No function {self._lprof_all_fname!r} defined "
                        "after AST transform") from None
                prof.add_function(func)
                # This method fetches the `LineProfiler.print_stats()`
                # output before the `os.unlink()` below happens
                run = self._run_and_profile(prof, parsed, prof.runcall, func)
        finally:
            os.unlink(tf.name)
        if "t" in parsed.opts:
            # I know it seems redundant to include this because users
            # could just use -r to get the info, but see the docstring
            # for why -t is included anyway.
            ip.user_ns["_total_time_taken"] = run.time_elapsed

        return self._handle_end(prof, run)

    _lprof_all_fname = "_lprof_cell"  # type: ClassVar[str]
