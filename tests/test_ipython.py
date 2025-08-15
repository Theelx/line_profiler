import unittest


class TestIPython(unittest.TestCase):
    def test_init(self):
        """
        CommandLine:
            pytest -k test_init -s -v
        """
        try:
            from IPython.testing.globalipapp import get_ipython
        except ImportError:
            import pytest
            pytest.skip()

        ip = get_ipython()
        ip.run_line_magic('load_ext', 'line_profiler')
        ip.run_cell(raw_cell='def func():\n    return 2**20')
        lprof = ip.run_line_magic('lprun', '-r -f func func()')

        timings = lprof.get_stats().timings
        self.assertEqual(len(timings), 1)  # 1 function

        func_data, lines_data = next(iter(timings.items()))
        print(f'func_data={func_data}')
        print(f'lines_data={lines_data}')
        self.assertEqual(func_data[1], 1)  # lineno of the function
        self.assertEqual(func_data[2], "func")  # function name
        self.assertEqual(len(lines_data), 1)  # 1 line of code
        self.assertEqual(lines_data[0][0], 2)  # lineno
        self.assertEqual(lines_data[0][1], 1)  # hits

    def test_lprun_all_autoprofile(self):
        try:
            from IPython.testing.globalipapp import get_ipython
        except ImportError:
            import pytest
            pytest.skip()
        
        ip = get_ipython()
        ip.run_line_magic('load_ext', 'line_profiler')
        lprof = ip.run_cell_magic(
            'lprun_all', line='-r', cell=self.lprun_all_cell_body)
        timings = lprof.get_stats().timings
        
        # 2 scopes: the module scope and an inner scope (Test.test)
        self.assertEqual(len(timings), 2)

        timings_iter = iter(timings.items())
        func_1_data, lines_1_data = next(timings_iter)
        func_2_data, lines_2_data = next(timings_iter)
        print(f'func_1_data={func_1_data}')
        print(f'lines_1_data={lines_1_data}')
        self.assertEqual(func_1_data[1], 1)  # lineno of the module
        self.assertEqual(len(lines_1_data), 2)  # only 2 lines were executed in this outer scope
        self.assertEqual(lines_1_data[0][0], 1)  # lineno
        self.assertEqual(lines_1_data[0][1], 1)  # hits
        
        print(f'func_2_data={func_2_data}')
        print(f'lines_2_data={lines_2_data}')
        self.assertEqual(func_2_data[1], 2)  # lineno of the inner function
        self.assertEqual(len(lines_2_data), 5)  # only 5 lines were executed in this inner scope
        self.assertEqual(lines_2_data[1][0], 4)  # lineno
        self.assertEqual(lines_2_data[1][1], self.loops - 1)  # hits

        # Check that the code is executed in the right scope and with
        # the expected side effects
        self.assertTrue(isinstance(ip.user_ns.get("Test"), type))
        self.assertTrue('z' in ip.user_ns)
        self.assertTrue(ip.user_ns['z'] is None)
    
    def test_lprun_all_autoprofile_toplevel(self):
        try:
            from IPython.testing.globalipapp import get_ipython
        except ImportError:
            import pytest
            pytest.skip()
        
        ip = get_ipython()
        ip.run_line_magic('load_ext', 'line_profiler')
        lprof = ip.run_cell_magic(
            'lprun_all', line='-r -p', cell=self.lprun_all_cell_body)
        timings = lprof.get_stats().timings
        
        # 1 scope: the module scope
        self.assertEqual(len(timings), 1)

        timings_iter = iter(timings.items())
        func_data, lines_data = next(timings_iter)
        print(f'func_data={func_data}')
        print(f'lines_data={lines_data}')
        self.assertEqual(func_data[1], 1)  # lineno of the module
        self.assertEqual(len(lines_data), 2)  # only 2 lines were executed in this outer scope
        self.assertEqual(lines_data[0][0], 1)  # lineno
        self.assertEqual(lines_data[0][1], 1)  # hits

        # Check that the code is executed in the right scope and with
        # the expected side effects
        self.assertTrue(isinstance(ip.user_ns.get("Test"), type))
        self.assertTrue('z' in ip.user_ns)
        self.assertTrue(ip.user_ns['z'] is None)
    
    def test_lprun_all_timetaken(self):
        try:
            from IPython.testing.globalipapp import get_ipython
        except ImportError:
            import pytest
            pytest.skip()
            
        ip = get_ipython()
        ip.run_line_magic('load_ext', 'line_profiler')
        ip.run_cell_magic('lprun_all', line='-t', cell=self.lprun_all_cell_body)

        # Check that the code is executed in the right scope and with
        # the expected side effects
        self.assertTrue(isinstance(ip.user_ns.get("Test"), type))
        self.assertTrue('z' in ip.user_ns)
        self.assertTrue(ip.user_ns['z'] is None)
        # Check that the elapsed time is written to the right scope
        self.assertTrue(ip.user_ns.get("_total_time_taken", None) is not None)

    # This example has 2 scopes
    # - The top level (module) scope, and
    # - The inner `Test.test()` (method) scope
    # when the `-p` flag is passed, the inner level shouldn't be
    # profiled
    loops = 20000
    lprun_all_cell_body = f"""
    class Test:
        def test(self):
            loops = {loops}
            for x in range(loops):
                y = x
                if x == (loops - 2):
                    break
    z = Test().test()
    """
