from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '0.1.1'


# Define an easy function for calling the shell
def call_shell(args):
    import subprocess
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return p.returncode, out, err

class BadExitStatus(Exception):
    pass

# Call the pkg-config command and either return a list of flags, or a list with the '-L' flag parts removed.
def call_pkgconfig(args, remove_flags=False):
    code, out, _ = call_shell(['pkg-config'] + args)
    if code != 0:
        raise BadExitStatus()
    flags = out.decode().strip().split(' ')
    if len(flags) == 1 and flags[0] == '':
        return []
    if remove_flags:
        return [ f[2:] for f in flags if len(f) > 2 ]
    else:
        return flags

# Try to find GSL with pkg-config
def find_gsl(libraries=[], library_dirs=[], extra_link_args=[], include_dirs=[], extra_compile_args=[]):
    default = dict(
        libraries = ['gsl', 'gslcblas', 'm'] + libraries,
        library_dirs=library_dirs,
        extra_link_args=extra_link_args,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args
        )
    try:
        result = dict(
            libraries = call_pkgconfig(['--libs-only-l', 'gsl'], remove_flags=True) + libraries,
            library_dirs = call_pkgconfig(['--libs-only-L', 'gsl'], remove_flags=True) + library_dirs,
            extra_link_args = call_pkgconfig(['--libs-only-other', 'gsl']) + extra_link_args,
            include_dirs = call_pkgconfig(['--cflags-only-I', 'gsl'], remove_flags=True) + include_dirs,
            extra_compile_args = call_pkgconfig(['--cflags-only-other', 'gsl']) + extra_compile_args
            )
    except OSError:
        return default
    except BadExitStatus:
        return default

    # remove empty args
    result = { key: value for key, value in result.items() if len(value) > 0 }
    return result

##
## Most of this is taken from the pybind11 python_example
##

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import pybind11
        return pybind11.get_include()

expint_src_files = [
    'expint.cpp', 
    'expint_asymp.cpp', 
    'expint_ei.cpp', 
    'expint_series.cpp']
cpg_src_files = [
    'cpg.cpp', 
    'cpg_pybind.cpp']
src_files = \
    [ os.path.join('libs', 'expint', 'src', fname) for fname in expint_src_files ] + \
    [ os.path.join('src', fname) for fname in cpg_src_files ]

ext_modules = [
    Extension(
        'cpg_likelihood.llh',
        # Sort input source files to ensure bit-for-bit reproducible builds
        # (https://github.com/pybind/python_example/pull/53)
        sorted(src_files),
        language='c++',
        **find_gsl(
            include_dirs=[
                # Path to pybind11 headers
                get_pybind_include(),
                'inc',
                os.path.join('libs', 'expint', 'inc'),
                os.path.join('libs', 'eigen')
            ],
            libraries = [],
            extra_compile_args=[
                '-O3',
                '-march=native',
                '-mtune=native',
                # Fast maths options
                '-ftree-vectorize',
                '-fno-math-errno',
                '-fassociative-math',
                '-fno-signed-zeros',
                '-fno-trapping-math',
                '-fno-rounding-math',
                '-fno-signaling-nans',
                '-fexcess-precision=fast',
                # Eigen disable debug
                '-DNDEBUG'
            ]
        )
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++14 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        build_ext.build_extensions(self)


setup(
    name='cpg_likelihood',
    version=__version__,
    author='ghcollin',
    author_email='',
    #url='https://github.com/pybind/python_example',
    description='Compound Poisson Generator likelihood function',
    long_description='',
    packages=['cpg_likelihood'],
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['numpy'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)