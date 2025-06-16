from setuptools import setup, find_packages

setup(
    name="mrvar",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "nibabel",
        "scipy"
    ],
    entry_points={
        'console_scripts': [
            'process-variance=pyqa.process_variance:main',
        ],
    },
) 