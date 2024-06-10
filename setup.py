from distutils.core import setup

setup(
    name='qlime',
    version='0.1',
    install_requires=[
        'numpy',
        'scikit-learn',
        'qiskit',
        'matplotlib'
    ],
    packages=[''],
    url='https://github.com/lirandepira/qlime',
    license='Apache 2.0',
    author='lpira',
    author_email='',
    python_requires='>=3.6',
    description='QLIME (Quantum Local Interpretable Model-agnostic Explanations) is a Python package for interpreting quantum neural networks using local surrogate models.'
)