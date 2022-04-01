import inspect
import os.path as op
import re
import sys
from unittest import SkipTest
import warnings

from pkgutil import walk_packages
from inspect import getsource

from numpydoc import docscrape
import celer

# copied from sklearn.fixes
if hasattr(inspect, 'signature'):  # py35
    def _get_args(function, varargs=False):
        params = inspect.signature(function).parameters
        args = [key for key, param in params.items()
                if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)]
        if varargs:
            varargs = [param.name for param in params.values()
                       if param.kind == param.VAR_POSITIONAL]
            if len(varargs) == 0:
                varargs = None
            return args, varargs
        else:
            return args
else:
    def _get_args(function, varargs=False):
        out = inspect.getargspec(function)  # args, varargs, keywords, defaults
        if varargs:
            return out[:2]
        else:
            return out[0]


public_modules = [
    # the list of modules users need to access for all functionality
    'celer',
    'celer.datasets'
]


def get_name(func):
    """Get the name."""
    parts = []
    module = inspect.getmodule(func)
    if module:
        parts.append(module.__name__)
    if hasattr(func, 'im_class'):
        parts.append(func.im_class.__name__)
    parts.append(func.__name__)
    return '.'.join(parts)


# functions to ignore args / docstring of
_docstring_ignores = [
    "celer.dropin_sklearn.Lasso.path",
    "celer.dropin_sklearn.path",
    "celer.dropin_sklearn.LassoCV.path",
    "celer.dropin_sklearn.MultitaskLasso.fit",
    "celer.dropin_sklearn.fit",
]
_tab_ignores = []


def check_parameters_match(func, doc=None):
    """Check docstring, return list of incorrect results."""
    incorrect = []
    name_ = get_name(func)
    if not name_.startswith('celer.'):
        return incorrect
    if inspect.isdatadescriptor(func):
        return incorrect
    args = _get_args(func)
    # drop self
    if len(args) > 0 and args[0] == 'self':
        args = args[1:]

    if doc is None:
        with warnings.catch_warnings(record=True) as w:
            try:
                doc = docscrape.FunctionDoc(func)
            except Exception as exp:
                incorrect += [name_ + ' parsing error: ' + str(exp)]
                return incorrect
        if len(w):
            raise RuntimeError('Error for %s:\n%s' % (name_, w[0]))
    # check set
    param_names = [name for name, _, _ in doc['Parameters']]
    # clean up some docscrape output:
    param_names = [name.split(':')[0].strip('` ') for name in param_names]
    param_names = [name for name in param_names if '*' not in name]
    if len(param_names) != len(args):
        bad = str(sorted(list(set(param_names) - set(args)) +
                         list(set(args) - set(param_names))))
        if not any(re.match(d, name_) for d in _docstring_ignores) and \
                'deprecation_wrapped' not in func.__code__.co_name:
            incorrect += [name_ + ' arg mismatch: ' + bad]
    else:
        for n1, n2 in zip(param_names, args):
            if n1 != n2:
                incorrect += [name_ + ' ' + n1 + ' != ' + n2]
    return incorrect


# TODO: readd numpydoc
# @requires_numpydoc
def test_docstring_parameters():
    """Test module docstring formatting."""
    public_modules_ = public_modules[:]

    incorrect = []
    for name in public_modules_:
        with warnings.catch_warnings(record=True):  # traits warnings
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        for cname, cls in classes:
            if cname.startswith('_'):
                continue
            with warnings.catch_warnings(record=True) as w:
                cdoc = docscrape.ClassDoc(cls)
            if len(w):
                raise RuntimeError('Error for __init__ of %s in %s:\n%s'
                                   % (cls, name, w[0]))
            if hasattr(cls, '__init__'):
                incorrect += check_parameters_match(cls.__init__, cdoc)
            for method_name in cdoc.methods:
                method = getattr(cls, method_name)
                incorrect += check_parameters_match(method)
            if hasattr(cls, '__call__'):
                incorrect += check_parameters_match(cls.__call__)
        functions = inspect.getmembers(module, inspect.isfunction)
        for fname, func in functions:
            if fname.startswith('_'):
                continue
            incorrect += check_parameters_match(func)
    msg = '\n' + '\n'.join(sorted(list(set(incorrect))))
    if len(incorrect) > 0:
        raise AssertionError(msg)


def test_tabs():
    """Test that there are no tabs in our source files."""
    ignore = _tab_ignores[:]

    for importer, modname, ispkg in walk_packages(celer.__path__,
                                                  prefix='celer.'):
        if not ispkg and modname not in ignore:
            # mod = importlib.import_module(modname)  # not py26 compatible!
            try:
                with warnings.catch_warnings(record=True):  # traits
                    __import__(modname)
            except Exception:  # can't import properly
                continue
            mod = sys.modules[modname]
            try:
                source = getsource(mod)
            except IOError:  # user probably should have run "make clean"
                continue
            assert '\t' not in source, ('"%s" has tabs, please remove them '
                                        'or add it to the ignore list'
                                        % modname)


documented_ignored_mods = tuple()
documented_ignored_names = """
""".split('\n')


def test_documented():
    """Test that public functions and classes are documented."""
    public_modules_ = public_modules[:]

    doc_file = op.abspath(op.join(op.dirname(__file__), '..', '..', 'doc',
                                  'api.rst'))
    if not op.isfile(doc_file):
        raise SkipTest('Documentation file not found: %s' % doc_file)
    known_names = list()
    with open(doc_file, 'rb') as fid:
        for line in fid:
            line = line.decode('utf-8')
            if not line.startswith('  '):  # at least two spaces
                continue
            line = line.split()
            if len(line) == 1 and line[0] != ':':
                known_names.append(line[0].split('.')[-1])
    known_names = set(known_names)

    missing = []
    for name in public_modules_:
        with warnings.catch_warnings(record=True):  # traits warnings
            module = __import__(name, globals())
        for submod in name.split('.')[1:]:
            module = getattr(module, submod)
        classes = inspect.getmembers(module, inspect.isclass)
        functions = inspect.getmembers(module, inspect.isfunction)
        checks = list(classes) + list(functions)
        for name, cf in checks:
            if not name.startswith('_') and name not in known_names:
                from_mod = inspect.getmodule(cf).__name__
                if (from_mod.startswith('celer') and
                        from_mod not in documented_ignored_mods and
                        name not in documented_ignored_names):
                    missing.append('%s (%s.%s)' % (name, from_mod, name))
    if len(missing) > 0:
        raise AssertionError('\n\nFound new public members missing from '
                             'doc/python_reference.rst:\n\n* ' +
                             '\n* '.join(sorted(set(missing))))
