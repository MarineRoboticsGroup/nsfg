from distutils.core import setup

setup(
    name='nsfg',
    version='',
    packages=['slam', 'stats', 'utils', 'factors', 'sampler', 'geometry'],
    install_requires=['arviz','dynesty','matplotlib','numpy','pandas','pymc3','scipy','Theano','TransportMaps'],
    package_dir={'': 'src'},
    url='',
    license='',
    author='',
    author_email='',
    description=''
)
