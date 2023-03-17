from setuptools import setup

from malenia import __version__

setup(
    name='malenia',
    version=__version__,

    url='https://github.com/RafaAyGar/malenia',
    author='Rafael Ayllón Gavilán',
    author_email='rafaaylloningeniero@gmail.com',

    py_modules=['malenia'],

    entry_points = {
        'console_scripts': [
            'malenia_check_condor_errors = malenia:check_condor_errors',
            'malenia_test = malenia:test',
        ],            
    },
)