from LDA import LDA

doc1 = ['ball','ball','ball','galaxy']
doc2 = ['galaxy','planets','referendum','policy']
doc3 = ['policy','referendum','policy','referendum']
doc4 = ['planets','galaxy','galaxy','ball']

corpus = [doc1, doc2, doc3, doc4]
alpha = 0.01
beta = 0.001
num_topics = 3

test_model = LDA(corpus, num_topics, alpha, beta)

test_model.train(100)