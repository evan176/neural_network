# This is for practice neural network

## Example
```
>>> from nn import NNClassifier
>>> clf = NNClassifier(
        hidden_layer_sizes=(6, 6), activation='sigmoid',
        solver='sgd', learning_rate=0.5, alpha=0.9,
        beta=0.999, max_iter=200, batch_size=50
    )
>>> clf.fit(training_data, training_label)
>>> predicted = clf.predict(test_data)
```

# References
- http://ufldl.stanford.edu/tutorial/
- http://www.slideshare.net/ckmarkohchang/tensorflow-60645128
- https://cs.stanford.edu/~quocle/tutorial1.pdf
- https://cs.stanford.edu/~quocle/tutorial2.pdf
- https://www.cs.cmu.edu/afs/cs/academic/class/15883-f15/slides/backprop.pdf
- https://github.com/yusugomori/DeepLearning
