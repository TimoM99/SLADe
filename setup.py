from setuptools import setup, find_packages

setup(
    name='slade',
    version='0.1.0',
    description='Semi-supervised techniques for learning from soft labels for anomaly detection use cases.',
    author='Timo Martens',
    author_email='timo.martens@kuleuven.be',
    url='https://github.com/TimoM99/SLADe/tree/main',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'anomatools>=3.0.3',
        'numpy>=1.24.3',
        'pyod>=2.0.1',
        'scikit_learn>=1.3.0',
        'scipy>=1.13.1',],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',],
    python_requires='>=3.8',
)