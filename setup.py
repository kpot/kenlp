# this file was created by following
# https://github.com/pypa/sampleproject/blob/master/setup.py
# as example

from setuptools import setup, find_packages

setup(
    name='kelp',
    # This allows to use git/hg to auto-generate new versions
    use_scm_version={"root": ".", "relative_to": __file__},
    setup_requires=['setuptools_scm'],
    description=('A set of natural language processing tools '
                 'written using Keras'),
    url='https://github.com/kpot/kelp',
    author='Kirill Mavreshko',
    author_email='kimavr@gmail.com',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],

    keywords='development',

    packages=find_packages(where='.', exclude=['tests']),
    # package_dir={'kelp': 'kelp'},
    install_requires=['Keras>=2.0.8', 'PyICU', 'polyglot'],
    tests_require=['pytest'],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={
    #     'console_scripts': [
    #         'kelp-mimic-train=kelp.mimic:main',
    #     ],
    # },
)
