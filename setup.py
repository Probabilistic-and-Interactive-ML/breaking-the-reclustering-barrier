from setuptools import find_packages, setup

setup(
    name="brb",
    version="0.1",
    description='This is the environment for running the codebase accompanying our paper on "Breaking the Reclustering Barrier" (BRB). To summarize, BRB prevents early performance plateaus in centroid-based deep clustering by periodically applying a soft reset to the feature encoder with subsequent reclustering. This allows the model to escape local minima and continue learning. We show that BRB significantly improves the performance of centroid-based deep clustering algorithms on various datasets and tasks.',
    author="Miklautz, Lukas and Klein, Timo and Sidak, Kevin and Tschiatschek, Sebastian and Plant, Claudia",
    author_email="your.email@example.com",  # TODO: Update this
    url="https://github.com/yourusername/yourproject",  # TODO: Update this
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.9",
    install_requires=[
        "anyio==4.4.0",
        "argon2-cffi==23.1.0",
        "arrow==1.3.0",
        "attrs==24.2.0",
        "babel==2.14.0",
        "beautifulsoup4==4.12.3",
        "bleach==6.1.0",
        "cffi==1.17.0",
        "charset-normalizer==3.3.2",
        "click==8.1.7",
        "colorama==0.4.6",
        "contourpy==1.3.0",
        "cycler==0.12.1",
        "debugpy==1.8.5",
        "decorator==5.1.1",
        "defusedxml==0.7.1",
        "docker-pycreds==0.4.0",
        "docstring_parser==0.16",
        "einops==0.8.0",
        "entrypoints==0.4",
        "exceptiongroup==1.2.2",
        "executing==2.0.1",
        "ffmpeg==1.4",
        "filelock==3.13.1",
        "fonttools==4.53.1",
        "frozendict==2.4.4",
        "gitpython==3.1.43",
        "h11==0.14.0",
        "httpcore==1.0.5",
        "httpx==0.27.2",
        "idna==3.8",
        "importlib-metadata==8.4.0",
        "importlib_resources==6.4.4",
        "ipdb==0.13.13",
        "ipykernel==6.29.5",
        "ipython==8.27.0",
        "jedi==0.19.1",
        "jinja2==3.1.4",
        "joblib==1.4.2",
        "jsonschema==4.23.0",
        "kiwisolver==1.4.5",
        "markdown-it-py==3.0.0",
        "markupsafe==2.1.5",
        "matplotlib==3.9.2",
        "mistune==3.0.2",
        "nbclient==0.10.0",
        "nbconvert==7.16.4",
        "nbformat==5.10.4",
        "nest-asyncio==1.6.0",
        "networkx==3.3",
        "nltk==3.9.1",
        "numba==0.60.0",
        "numpy==1.26.4",
        "packaging==24.1",
        "pandas==2.2.2",
        "parso==0.8.4",
        "patsy==0.5.6",
        "pexpect==4.9.0",
        "pickleshare==0.7.5",
        "pillow==10.4.0",
        "pip==24.2",
        "platformdirs==4.2.2",
        "prometheus_client==0.20.0",
        "prompt-toolkit==3.0.47",
        "protobuf==4.25.3",
        "psutil==6.0.0",
        "pygments==2.18.0",
        "pyparsing==3.1.4",
        "pysocks==1.7.1",
        "python-dateutil==2.9.0",
        "fastjsonschema==2.20.0",
        "python-json-logger==2.0.7",
        "pynvml==11.5.3",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "pytz==2024.1",
        "pyyaml==6.0.2",
        "pyzmq==26.2.0",
        "referencing==0.35.1",
        "regex==2024.7.24",
        "requests==2.32.3",
        "rich==13.7.1",
        "ruff==0.6.2",
        "scikit-learn==1.2.2",
        "scikit-learn-extra==0.3.0",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "send2trash==1.8.3",
        "setuptools==72.2.0",
        "six==1.16.0",
        "soupsieve==2.5",
        "stack_data==0.6.2",
        "statsmodels==0.14.2",
        "sympy==1.13.2",
        "shortuuid==1.0.13",
        "terminado==0.18.1",
        "threadpoolctl==3.5.0",
        "tinycss2==1.3.0",
        "tqdm==4.66.5",
        "traitlets==5.14.3",
        "typing-extensions==4.12.2",
        "tyro==0.8.10",
        "umap-learn==0.5.6",
        "urllib3==2.2.2",
        "wandb==0.16.6",
        "wcwidth==0.2.13",
        "webcolors==24.8.0",
        "webencodings==0.5.1",
        "websocket-client==1.8.0",
        "wheel==0.44.0",
        "zipp==3.20.1",
        "zstandard==0.23.0",
    ],
)
