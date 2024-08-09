
# Pre Processing Pipeline

Here we provide the code for pre-processing commits in order to create machine readable vectors.

## Introduction



We have organized the replication package into two folders and five Python files:

1. data: This folder contains all the data stored and required to run the pre processing pipeline.
	- **Commit-logs:** Contains commit log information for each cloned project. To extract this data, use the bashscripts/git-log2json.sh script.
	- **Commit-stats:** Contains commit log code statistics for each cloned project. To extract this data, use the bashscripts/git-stat2json.sh script.
	- **Code metrics:** This folder contains code metrics extracted using the Understand tool for each project.
			To get code metrics for each project, please use and follow the instructions of the Understand tool (https://scitools.com/).
	- **Refactorings:** This folder contains refactorings extracted using PyRef.
			To get code refactorings for each project, please use and follow the instructions of the PyRef tool (https://github.com/PyRef/PyRef/).

Our pre processing code pipeline is based on the following packages and versions:
- nltk: 3.8.1
- pyenchant: 3.2.2
- textstat: 0.7.4

The following code can be used to install all packages in the environment.
```bash
  pip install -r requirements.txt
```
To load the dataset unzip the data.zip file in the root directory of the project. You can use the command below:
```bash
  unzip data.zip
```

We recommend using Python version Python 3.10.12 and every Python requirement should be met.

  run pipeline.py to start the pipeline for one sample project
```bash
  python3 pipeline.py
```


