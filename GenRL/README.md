# Generative Relation Linking (GenRL)
This folder contains the implementation for the method described in the paper [Generative Relation Linking for Question Answering over Knowledge Bases](https://arxiv.org/abs/2108.07337) accepted at ISWC 2021.

### Knowledge Integration
The knowledge integration component aims to enrich the encoder representation with information related to the entities recognized in the questions, such as entity types and candidate relations from the target Knowledge Base.

The output of this step is available for each dataset:
- QALD-9 (DBpedia): [data/qald9/qald9_test.json](https://github.com/IBM/kbqa-relation-linking/blob/master/GenRL/data/qald9/qald9_test.json)
- LC-QuAD 1.0 (DBpedia): [data/lcquad1/lcquad1_test.json](https://github.com/IBM/kbqa-relation-linking/blob/master/GenRL/data/lcquad1/lcquad1_test.json)
- LC-QuAD 2.0 (Wikidata): [data/lcquad2/lcquad2_test.json](https://github.com/IBM/kbqa-relation-linking/blob/master/GenRL/data/lcquad2/lcquad2_test.json)
- Simple Questions (Wikidata): [data/simpleq/simpleq_test.json](https://github.com/IBM/kbqa-relation-linking/blob/master/GenRL/data/simpleq/simpleq_test.json)

### Sequence-to-Sequence
```
pip install -U -r requirements.txt
```

QALD-9
```
python seq2seq.py --test_file data/qald9/qald9_test.json --model_name gaetangate/bart-large_genrl_qald9 --device cuda --output data/qald9/qald9_test_GenRL.json
```

LC-QuAD 1.0
```
python seq2seq.py --test_file data/lcquad1/lcquad1_test.json --model_name gaetangate/bart-large_genrl_lcquad1 --device cuda --output data/lcquad1/lcquad1_test_GenRL.json
```

LC-QuAD 2.0
```
python seq2seq.py --test_file data/lcquad1/lcquad2_test.json --model_name gaetangate/bart-large_genrl_lcquad2 --device cuda --output data/lcquad1/lcquad2_test_GenRL.json
```

Simple Questions
```
python seq2seq.py --test_file data/simpleq/simpleq_test.json --model_name gaetangate/bart-large_genrl_simpleq --device cuda --output data/simpleq/simpleq_test_GenRL.json
```

### Knowledge Validation

In order to run the knowledge validation, the users will need SPARQL endpoints loaded correct with version of datasets: QALD-9 (DBpedia 2016-10), LC-QuAD 1.0 (DBpedia 2016-04), LC-QuAD 2.0 (Wikidata), and Simple Questions (Wikidata). 

QALD-9
```
python knowledge_val.py --model_output data/qald9/qald9_test_GenRL.json --val_cache qald9_cache.json --sparql_endpoint $dbpedia_2016_10$ --val_output data/qald9/qald9_test_validated.json
```

LC-QuAD 1.0
```
python knowledge_val.py --model_output data/lcquad1/lcquad1_test_GenRL.json --val_cache lcquad1_cache.json --sparql_endpoint $dbpedia_2016_04$ --val_output data/lcquad1/lcquad1_test_validated.json
```

LC-QuAD 2.0
```
python knowledge_val.py --model_output data/lcquad2/lcquad1_test_GenRL.json --val_cache lcquad1_cache.json --sparql_endpoint $wikidata_endpoint$  --val_output data/qald9/lcquad1_test_validated.json
```

Simple Questions
```
python knowledge_val.py --model_output data/simpleq/simpleq_test_GenRL.json --val_cache simpleq_cache.json --sparql_endpoint $wikidata_endpoint$ --val_output data/simpleq/simpleq_test_validated.json
```

### Evaluation
