# toyLDA
This is a toy implementation of an Latent Dirichlet Allocation model for Topic modeling.


## How to use:

In main.py, there is an example of the model. The model outputs a reassignment to the topics. Alpha represents document-topic density and that means that higher values of alpha means that documents are more likely to be assigned more than one topic, lower values of alpha means that documents are more likely to be assigned a single topic. Beta represents a topic-word density. Similar to alpha, higher beta means topics are more likely to be many words and lower beta means topics are more likely to be individual words.

## Credits:

I would like to include that this implementation is fully based on the Latent Dirichlet Allocation paper by David M. Blei, Andrew Y. Ng, and Michael I. Jordan.
The paper can be found at the link below:
http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
