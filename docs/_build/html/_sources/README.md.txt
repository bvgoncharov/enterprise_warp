# Documentation how-to

Web pages are designed in .rst files.

Compilation is done through "Readthedocs" account online, automatically.

To perform build/compilation locally, Sphinx must be installed. Compilation with Sphinx-2.1.2 did not work. I had to do: `pip install sphinx==1.4.8 --user`. On my OzStar account, Sphinx is installed with Python 3.

Then, one has to run `make html`.
