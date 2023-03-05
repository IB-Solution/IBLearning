import PyModuleGenerator

long_description = open("D:\Dev\Projects\IBLearning\README.md", "r", encoding="utf-8").read()

config: PyModuleGenerator.PyModuleGeneratorConfig = PyModuleGenerator.PyModuleGeneratorConfig(
    pythonCommand="py",    # Python command to use (python or python3 or py)

    modulePath="D:\Dev\Projects\IBLearning\IBLearning",
    buildFolder="D:\Dev\Projects\IBLearning\\build",
    moduleName="IBLearning",
    moduleVersion="1.0.3",
    moduleDescription="IBLearning is a Python module that contains a set of tools to help you to make your own machine learning algorithms.",
    moduleLongDescription=long_description,
    moduleLongDescriptionType="text/markdown",

    githubURL="https://github.com/IB-Solution/IBLearning",
    moduleAuthor="Alix Hamidou",
    moduleAuthorEmail="alix.hamidou@gmail.com",
    moduleLicense="MIT",

    moduleDependencies=[
        "nltk>=1.1.0"
    ],
    moduleTags=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic"
    ] # https://pypi.org/classifiers/
)


PyModuleGenerator.PyModuleGenerator(
    config=config,
    clearBuildFolder=True,      # Erase the build folder after the build
    publishToPypi=True          # Publish the module to pypi
)