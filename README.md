# Research
## Deep-Residual-Learning-for-Image-Recognition

The authors of the *Deep Residual Learning* (DRL) paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), describe a method for training a network to recognize images. The method is called *Residual Networks* (ResNets) and is a generalization of the *Convolutional Neural Network* (CNN) model.

At the time of writing the paper, computer vision suffered a problem where as you made CNNs larger, their performance would suffer. The large model would perform worse in both train and test datasets, which meant the model wasn't overfitting to the train data. A model with 18 layers would perform better than one with 34 layers. The researchers found this logically flawed.

Think of each layer of the neural network as fine-tuning a function that maps to the given dataset. As you add more layers, you add more definition to the function making it more specific to the dataset. Any function created by a small model must be *creatable* by a large model. Essentially if the additional CNN layers in the large model just learned to not change the outputs of the small model, it would achieve the same result (this is what they call the *identity function*).

But why weren't the large models learning the identity function now? This has to do with how models are initialized. The model is initialized with random weights centered around a mean of 0. This means that as the weights change, they tend to the value 0. If the identity function requires weights further away, it would take longer to learn. The problem now was figuring out how to make the model biased toward the identity function.

To solve this, the researchers used the natural properties of the weights. Since the weights were tending toward 0 anyway, if you add then to the identity function, the output will tend toward the identity.

Consider the output of a CNN layer with input $x$ and output $H(x)$
$$H(x)=x+f(x)$$
Where $f(x)$ is the change in the input to get the output. The researchers hypothesized that if a model is able to get $H(x)$ from $x$, then it should also be able to get $f(x)$ from $x$. So they set up the convolutional layers to output $f(x)$, a tensor the same shape as $x$. Since the weights tend to 0, we know that $f(x)$ will also tend to 0. By adding the identity function, if $f(x)$ tends to 0, it'll instead tend to the identity ($x$).

Original Output: $f(x)$
Residual Output: $f(x)+x$

Residual layers simply add the input of a *block* (multiple feed-forward layers) to the output of the block. This makes the deeper layers tend toward the identity meaning they make smaller changes to the output.