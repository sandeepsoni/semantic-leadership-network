Description of the pipeline
=====

If you want to use the code developed for the [cultural analytics paper](https://culturalanalytics.org/article/18841-abolitionist-networks-modeling-language-change-in-nineteenth-century-activist-newspapers), then you would need to follow the steps below:

* To train the embeddings model, use the Java code developed by David Bamman [in this repository](https://github.com/dbamman/geoSGLM). 
For the cultural analytics paper, we modified this Java code to emit output embeddings and include an additional facet.
This modification is available in the form of a pull request (as of 3/31/2022) [here](https://github.com/dbamman/geoSGLM/pull/3).

If you're interested in the processed data, i.e., the embeddings from the data used in the paper, it can all be downloaded from [this link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EWYMFG)

* Next, semantic changes are learned by running [`pipeline_temporal.sh`](https://github.com/sandeepsoni/semantic-leadership-network/blob/main/scripts/pipeline_temporal.sh)
* Next, the leadership stats are calculated by running [`pipeline_sources.sh`](https://github.com/sandeepsoni/semantic-leadership-network/blob/main/scripts/pipeline_sources.sh)
* Finally, the randomization experiments are calculated by running [`pipeline_randomization.sh`](https://github.com/sandeepsoni/semantic-leadership-network/blob/main/scripts/pipeline_randomization.sh)

Cite
====
If you use the processed data (embeddings from abolitionist newspaper corpus), please consider citing our data link as:
```
@data{DVN/EWYMFG_2021,
author = {Soni, Sandeep and Klein, Lauren and Eisenstein, Jacob},
publisher = {Harvard Dataverse},
title = {{Abolitionist Networks: Modeling Language Change in Nineteenth-Century Activist Newspapers}},
year = {2021},
version = {V1},
doi = {10.7910/DVN/EWYMFG},
url = {https://doi.org/10.7910/DVN/EWYMFG}
}
```
If you end up using the code from this repository, please also consider citing our paper as:

```
@article{soni2021abolitionist,
  title={Abolitionist Networks: Modeling Language Change in Nineteenth-Century Activist Newspapers},
  author={Soni, Sandeep and Klein, Lauren F and Eisenstein, Jacob},
  journal={Journal of Cultural Analytics},
  volume={6},
  number={1},
  pages={18841},
  year={2021},
  publisher={Department of Languages, Literatures, and Cultures}
}
```

Contact
====

Please contact Sandeep Soni (soni.sandeepb@gmail.com) or Lauren Klein (lklein@gmail.com) for any inquiries about the data or code.
