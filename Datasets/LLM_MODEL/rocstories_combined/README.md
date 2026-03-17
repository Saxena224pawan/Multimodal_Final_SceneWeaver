---
dataset_info:
  features:
  - name: title
    dtype: string
  - name: sentences
    sequence: string
  - name: shuffled_sentences
    sequence: string
  - name: gold_order
    sequence: int64
  splits:
  - name: train
    num_bytes: 54656181
    num_examples: 98161
  download_size: 32722430
  dataset_size: 54656181
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Dataset Card for Dataset Name

<!-- Provide a quick summary of the dataset. -->

This dataset is a merged version of the Spring 2016 and Winter 2017 versions of the [ROCStories](https://cs.rochester.edu/nlp/rocstories/) Dataset. You can request the dataset from 
using the form on the website as well. 

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->



- **Curated by:** Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, James Allen
- **Language(s) (NLP):** English

### Dataset Sources

<!-- Provide the basic links for the dataset. -->


- **Paper:** [A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories](https://arxiv.org/abs/1604.01696)


## Uses

<!-- Address questions around how the dataset is intended to be used. -->
Sentences Ordering, Sentence Comprehension, Evaluation of Language Models on Sentence Ordering. 



## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

The dataset contains 98161 stories, each containing 5 sentences. Each instance or story includes a title, the original and the shuffled order of the sentences and a list of integers as gold order for evaluating prediction from a model.



## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@misc{mostafazadeh2016corpus,
      title={A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories}, 
      author={Nasrin Mostafazadeh and Nathanael Chambers and Xiaodong He and Devi Parikh and Dhruv Batra and Lucy Vanderwende and Pushmeet Kohli and James Allen},
      year={2016},
      eprint={1604.01696},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```



## Dataset Card Authors

Shawon Ashraf
