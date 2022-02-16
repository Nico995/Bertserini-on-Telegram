import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bertserini_on_telegram",
    version="1.2",
    author="Nicola Occelli",
    author_email="nicola.occelli@studenti.polito.it",
    description="A library, based on PyTorch, that implements bertserini code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nico995/Bertserini-on-Telegram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "transformers",
        "pyserini",
        "telegram",
        "datasets",
        "faiss-gpu",
        "fasttext",
        "iso639",
        "jsonargparse"

    ]
)
