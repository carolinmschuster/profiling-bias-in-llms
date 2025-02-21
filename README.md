
# Profiling Bias in LLMs: Stereotype Dimensions in Contextual Word Embeddings

### Abstract

Large language models (LLMs) are the foundation of the current successes of artificial intelligence (AI), however, they are unavoidably biased. To effectively communicate the risks and encourage mitigation efforts these models need adequate and intuitive descriptions of their discriminatory properties, appropriate for all audiences of AI. We suggest bias profiles with respect to stereotype dimensions based on dictionaries from social psychology research. Along these dimensions we investigate gender bias in contextual embeddings, across contexts and layers, and generate stereotype profiles for twelve different LLMs, demonstrating their intuition and use case for exposing and visualizing bias.


***Please cite our paper when using this work for your project:***
~~~
@inproceedings{
schuster2024profiling,
title={Profiling Bias in {LLM}s: Stereotype Dimensions in Contextual Word Embeddings},
author={Carolin M. Schuster and Maria-Alexandra Dinisor and Shashwat Ghatiwala and Georg Groh},
booktitle={The Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies},
year={2024}
}
~~~

# Usage

Environment
```
conda env create -f profiling-env.yaml
conda activate profiling-env
```
Running the analysis for female and male associated names, with stereotype dimensions embedded by generated gender non-specific examples.
```
python projection.py google-bert/bert-base-uncased --populations names.json --examples generated_examples.txt
```
Running the analysis for gendered terms
```
python projection.py google-bert/bert-base-uncased --populations terms.json --examples generated_examples.txt
```
Running with access token for gated models on huggingface
```
python projection.py meta-llama/Meta-Llama-3-8B --populations names.json --examples generated_examples.txt --hf_token {YOUR_TOKEN}
```



## Resources


Dictionaries of the stereotype content model ([Nicolas, 2021](https://onlinelibrary.wiley.com/doi/abs/10.1002/ejsp.2724)), available from https://osf.io/yx45f/
- Seed dictionary: https://osf.io/ghfkb
- Full dictionary: https://osf.io/m9nb5

### Vocabulary populations

Female and male associated names: USA "Top Names Over the Last 100 Years", retrieved September 2024 from https://www.ssa.gov/oact/babynames/decades/century.html

Binary gendered terms: Combination of terms from WEAT experiments ([Caliskan, 2017](https://www.science.org/doi/abs/10.1126/science.aal4230)), Math vs Arts and Science vs Arts, italic terms excluded
- Female: female, woman, girl, sister, she, daughter,mother, aunt, grandmother, *hers*, *her*
- Male: male, man, boy, brother, he, son, father, uncle, grandfather, *his*, *him*


### Literature

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183-186.

Engler, J., Sikdar, S., Lutz, M., & Strohmaier, M. (2022, December). SensePOLAR: Word sense aware interpretability for pre-trained contextual word embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2022 (pp. 4607-4619).

Fraser, K. C., Nejadgholi, I., & Kiritchenko, S. (2021, August). Understanding and Countering Stereotypes: A Computational Approach to the Stereotype Content Model. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) (pp. 600-616).

Mathew, B., Sikdar, S., Lemmerich, F., & Strohmaier, M. (2020, April). The polar framework: Polar opposites enable interpretability of pre-trained word embeddings. In Proceedings of The Web Conference 2020 (pp. 1548-1558).

Nicolas, G., Bai, X., & Fiske, S. T. (2021). Comprehensive stereotype content dictionaries using a semi‐automated method. European Journal of Social Psychology, 51(1), 178-196.
