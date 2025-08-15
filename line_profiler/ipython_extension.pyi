from IPython.core.magic import Magics
from . import LineProfiler


class LineProfilerMagics(Magics):
    def parse_parameters(self, parameter_s: str) -> str:
        ...

    def lprun(self, parameter_s: str = ...) -> LineProfiler | None:
        ...

    def lprun_all(self,
                  parameter_s: str = "",
                  cell: str | None = None) -> LineProfiler | None:
        ...
