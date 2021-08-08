import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enterprise_warp",
    version="0.0.2",
    author="Boris Goncharov, Andrew Zic",
    author_email="goncharov.boris@physics.msu.ru",
    description="A wrapper for Nanograv's Enterprise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bvgoncharov/enterprise_warp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
