from setuptools import setup, find_packages

setup_requires = [
    ]

install_requires = [
    'torch==1.5.1',
    'torchvision==0.6.1',
    'transformers==4.30.0',
    'scipy==1.5.0']

setup(
    name='mercy_transformer',
    version='0.1',
    description='mercy_transformer',
    author='mercy',
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    entry_points={
        'console_scripts': [
            'publish = flowdas.books.script:main',
            'scan = flowdas.books.script:main',
            'update = flowdas.books.script:main',
            ],
        },
    )