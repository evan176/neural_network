try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='nn',
    description="",
    author="Evan Gui",
    author_email="evan176.gui@gmail.com",
    license='MIT',
    packages=['nn'],
    zip_safe=False,
    install_requires=[
        'numpy>=1.9.3'
    ],
    test_suite="pytest",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ]
)

