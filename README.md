# Learning Code-Query Interaction for Enhancing Code Searches

This repository includes the source code for the paper
'Learning Code-Query Interaction for Enhancing Code Searches'.



## Dependency

> Tested in Ubuntu 18.04

Install all the dependent packages via pip:

	$ pip install -r requirements.txt


## Dataset

### Download Datasets
we use two datasets:

1) CODEnn[1] could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing)
1) Cosbench[2] could be downloaded from [Google Drive](https://drive.google.com/file/d/1I5gimDYK7WaiGbSGnBO9jwT3txR1dHlY/view?usp=sharing)

Download the dataset and replace files in the `/data` folder. 


<!--### Data Format
The data format required by our model is 

```json
[   
    {
        "query" : "xxx",
        "name"  : "xxx",
        "body"  : "xxx"
    },
    ...
]
``` -->
The `/data/example` folder provides a small sample dataset for quick deploying.  

### Data Preparation
To generate preprocessed data:
```bash
python pipeline.py
```


## Train & Evaluate
  To train our model:
   ```bash
   python main.py --mode train
   ```   
   
  To evaluate our model:
   ```bash
   python main.py --mode eval
   ``` 
   
### References
[1] X. Gu, H. Zhang, and S. Kim, “Deep code search,” in 2018 IEEE/ACM
40th International Conference on Software Engineering (ICSE). IEEE,
2018, pp. 933–944.

[2] S. Yan, H. Yu, Y. Chen, B. Shen, and L. Jiang, “Are the code snippets
what we are searching for? a benchmark and an empirical study on
code search with natural-language queries,” in 2020 IEEE 27th International Conference on Software Analysis, Evolution and Reengineering
(SANER). IEEE, 2020, pp. 344–354.