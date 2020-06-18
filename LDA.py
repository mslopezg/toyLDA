import random
class LDA():

    corpus_topics = []

    def __init__(self, corpus, num_topics, alpha, beta):
        # Corpus is the body of documents - represented as a list of docs, each doc is a list of words
        # num_words is the number of words per document 
        # (Note: in the paper, a Poisson distribution can be used for documents with different number of words. In this example model, we assume all docs have the same number of words)
        # num_topics is the desired number of topics to be acquired
        # alpha is the dirichlet prior that is given to the distribution of topics given a document
        # beta is the dirichlet prior that is given to the distribution of words given a topic
        self.corpus = corpus
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta

    def initcorpus(self, num_docs, num_words):
        # Assign a random topic to each word
        initcorp = []
        for _ in range(num_docs):
            initdoc = []
            for _ in range(num_words):
                initdoc.append(random.randrange(0, self.num_topics, 1))
            initcorp.append(initdoc)

        return initcorp

    def topic_given_doc(self, doc_index, word_index):
        # Calculates the probability of a topic given a document 
        # the index that is passed in, is the word that is currently being reassigned a topic
        # The probability of a topic given a document, while ignoring the current word that we are updating is calculated as follows:
        # First, count the number of instances that each topic occurs in a single document.
        # Then, add the alpha dirichlet prior to every topic count
        # Finally, normalize and return the distribution
        dist_tgp = []
        count = 0
        # iterate through every topic and then iterate through every word in the document except the one we are updating
        # if the word has the topic increment the count then once every word has been scanned add it to the counted list
        for topic_num in range(self.num_topics):
            for i in range(len(self.corpus_topics[doc_index])):
                if(i != word_index):
                    if(topic_num == self.corpus_topics[doc_index][i]):
                        count += 1
            dist_tgp.append(count)
            count = 0
        
        # Add the dirichlet prior
        alpha_dist = [x + self.alpha for x in dist_tgp]

        # Normalize
        norm_factor = 0
        for i in alpha_dist:
            norm_factor += i
                    
        norm_alpha_dist = [x / norm_factor for x in alpha_dist]

        return norm_alpha_dist

    
    def word_given_topic(self, doc_index, word_index):
        # Calculates the probability of a word given a topic
        # the index that is passed in, is the word that is currently being reassigned a topic
        # the probability of a word given a topic is calculated as follows:
        # first count the number of instances that each word occurs in topic 0 throughout the corpus (not including the word that we are updating)
        # normalize the distribution
        # do so for every single topic
        # once you have every distribution of the words given the topic, retrieve the probability of the word we are updating from each distribution
        # so for example if the word is ball, each distribution will have the probability that ball occurs given a topic
        # place all the probabilities in the list

        list_wgt = []   # this list will be appended with the probability that the word we are updating occurs in topic 0,1,2 etc 

        # create a flat list of words and topics from the corpus
        flat_words = [item for sublist in self.corpus for item in sublist]
        flat_topics = [item for sublist in self.corpus_topics for item in sublist]
        flat_list = list(zip(flat_words,flat_topics))

        # create a list of the unique words in the corpus
        unique_list = list(set(flat_words))

        # calculate flat index
        flat_index = doc_index*len(self.corpus[0]) + word_index
        updating_word = flat_words[flat_index]

        # for every topic append the probability
        for topic_num in range(self.num_topics):
            # create a list of words with the current topic (not including the updating word)
            topic_word_list = [flat_list[i][0] for i in range(len(flat_list)) if(i != flat_index) if(flat_list[i][1] == topic_num)]
            # create an empty list for the counts of words in the corpus of the current topic
            word_counts = []
            for w in unique_list:
                count = topic_word_list.count(w)
                word_counts.append(count)
            # add the beta dirichlet prior to the word given topic distribution
            beta_dist = [x + self.beta for x in word_counts]
            # Normalize
            norm_factor = 0
            for i in beta_dist:
                norm_factor += i       
            norm_beta_dist = [x / norm_factor for x in beta_dist]
            # add probability of desired word given topic to the output
            list_wgt.append(norm_beta_dist[unique_list.index(updating_word)])

        return list_wgt

    def train(self, iter):
        # Train the model for 'iter' number of iterations.

        # Assuming every doc has the same number of words, retrieve the number of words
        num_words = len(self.corpus[0])
        # Retrieve the number of documents
        num_docs = len(self.corpus)

        # Initialize the corpus by assigning a random topic to each word
        self.corpus_topics = self.initcorpus(num_docs,num_words) 

        print("This is the corpus of words \n", self.corpus)
        print("This is the initial ~random~ assignment of topics to words: \n",self.corpus_topics, "\n")

        # Training step   
        for _ in range(iter):
            for doc_index in range(num_docs):
                for word_index in range(num_words):
                    # iterate through the entire corpus getting the indices of each word we are updating
                    # calculate topic given document distribution
                    # dist_tgp is a list of n topics
                    dist_tgp = self.topic_given_doc(doc_index, word_index)
                    # calculate probability of a word given the topic
                    # prob_wgt is a list of n probabilities for a word given the n topics
                    prob_wgt = self.word_given_topic(doc_index, word_index)

                    # create the topic distribution
                    topic_dist = [dist_tgp[i] * prob_wgt[i] for i in range(self.num_topics)]

                    # Normalize
                    norm_factor = 0
                    for i in topic_dist:
                        norm_factor += i
                    
                    norm_topic_dist = [x / norm_factor for x in topic_dist]

                    # Sample a new topic based on the distribution and update
                    sample_num = random.random()
                    cutoff = 0
                    for i in range(self.num_topics):
                        cutoff = cutoff + norm_topic_dist[i]
                        if (sample_num < cutoff):
                            # assign the word this new topic
                            self.corpus_topics[doc_index][word_index] = i
                            break
                    
        print("These are the newly assigned topics to each word.\n")
        print(self.corpus_topics)
