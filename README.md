# Multilingual Formality Classification

We provide code for models training and inference procedure for the paper "Detecting Text Formality: A Study of Text Classification Approaches".
All the models tested in this work are presented in the folder `notebooks`.


## Citation

To acknowledge our work, please, use the corresponding citation:

```
@inproceedings{dementieva-etal-2023-detecting,
    title = "Detecting Text Formality: A Study of Text Classification Approaches",
    author = "Dementieva, Daryna  and
      Babakov, Nikolay  and
      Panchenko, Alexander",
    editor = "Mitkov, Ruslan  and
      Angelova, Galia",
    booktitle = "Proceedings of the 14th International Conference on Recent Advances in Natural Language Processing",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.ranlp-1.31",
    pages = "274--284",
    abstract = "Formality is one of the important characteristics of text documents. The automatic detection of the formality level of a text is potentially beneficial for various natural language processing tasks. Before, two large-scale datasets were introduced for multiple languages featuring formality annotation{---}GYAFC and X-FORMAL. However, they were primarily used for the training of style transfer models. At the same time, the detection of text formality on its own may also be a useful application. This work proposes the first to our knowledge systematic study of formality detection methods based on statistical, neural-based, and Transformer-based machine learning methods and delivers the best-performing models for public usage. We conducted three types of experiments {--} monolingual, multilingual, and cross-lingual. The study shows the overcome of Char BiLSTM model over Transformer-based ones for the monolingual and multilingual formality classification task, while Transformer-based classifiers are more stable to cross-lingual knowledge transfer.",
}
```
