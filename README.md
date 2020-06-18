# toyLDA
This is a toy implementation of an Latent Dirichlet Allocation model for Topic modeling.

## Assumptions made:

I decided to make an assumption to simplify my implementation of the model. I assumed that any input of documents were going to have the same number of words. I address this assumption in a comment in the __init__ definition in the LDA class.

## How to use:

In main.py, there is an example of the model. The model outputs a reassignment to the topics. Alpha represents document-topic density and that means that higher values of alpha means that documents are more likely to be assigned more than one topic, lower values of alpha means that documents are more likely to be assigned a single topic. Beta represents a topic-word density. Similar to alpha, higher beta means topics are more likely to be composed of many words and lower beta means topics are more likely to be composed of individual words.

*Note: 
In my example for the model, it is likely that on the first run the topics are not necessarily correctly grouped. This can be attributed to the small number and size of the documents.

## Credits:

I would like to include that this implementation is fully based on the Latent Dirichlet Allocation paper by David M. Blei, Andrew Y. Ng, and Michael I. Jordan.
The paper can be found at the link below:

http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
