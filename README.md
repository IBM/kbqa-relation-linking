# KBQA Relation Linking

# Prerequisites
 Please download the following files into your machine before running the service. There paths are set in the configuration file.
 
 * GoogleNews-vectors-negative300.bin.gz,.the word2vec pre-trained Google News corpus (3 billion running words) word vector model. Please download it from [here](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).
 
 * glove_vocab.pkl file from [here](https://ibm.box.com/s/huox6kau50ceqanqso3z97g8lg9z0ho5)
 
 * OpenNRE trained model from [upload it somewhere](https://ibm.box.com/s/nd0v0ln1u5apl9uejac079pli1pz3p2p)
 
 The other data files that are used are provided in this repository itself. 
 
 The dependencies that are required to run the system are listed in `requirements.txt`.
 
 
 # Configuration 
 
The KBQA relation linking service is requires a configuration file that specifies the different data files and settings used by the service. `config` directory contains 
the configuration files for QALD-7, QALD-9, and LC-QuAD 1.0 experiments.

Please check the paths to files you downloaded are correctly set in the configuration.
 
 
 # Evaluation script 
 
The evaluation script is at `src/evaluation/local_evaluation.py`. Please set the working directory to `src` so that all
relative paths are resolved correctly.

It requires the two parameters:
 
* --config_path path to the configuration file to run the service.

* --input_path path to the input file containing the input to the system (i.e., the question text and the 
EAMR, the corresponding AMR graph with entity linking)

For the three experiments, these files can be found in `data\input` and `config` directories.
 
 # Publication 

 Please use the following paper to refer to this work.
 
 ```
@inproceedings{mihindu-sling-2020,
    title = "Leveraging Semantic Parsing for Relation Linking over Knowledge Bases",
    author = "Mihindukulasooriya, Nandana and Rossiello, Gaetano and Kapanipathi, Pavan and Abdelaziz, Ibrahim and Ravishankar, Srinivas and Yu, Mo and Gliozzo, Alfio and Roukos, Salim and Gray, Alexander",
    booktitle="The Semantic Web -- ISWC 2020",
    year="2020",
    publisher="Springer International Publishing",
    address="Cham",
    pages="402--419",
    url = "https://link.springer.com/chapter/10.1007/978-3-030-62419-4_23",
    doi = "10.1007/978-3-030-62419-4_23"
}
```
 
 # License
  
 This work is released under Apache 2.0 license. 
