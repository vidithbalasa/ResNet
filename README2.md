# Research
## Understanding the Paper : Deep Residual Learning

The authors of the *Deep Residual Learning* (DRL) paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), describe a method for training a network to recognize images. The method is called *Residual Networks* (ResNets) and is a generalization of the *Convolutional Neural Network* (CNN) model.

1. Models got better w/ size until they didn't
2. They weren't overfitting, they just couldn't learn the data (a degredation problem)

1. researchers realized the bigger models should just be able to learn the identity and not have worse performance
2. they realized the models couldn't because it would sometimes be too hard to learn the right weights
3. they solved this by adding a *skip connection* to the network


At the time of writing the paper, classical CNN models suffered an issue where model accuracy increases linearly with network depth (# of layers) until a certain point, where the accuracy starts degrading. The researchers called this the *degredation problem*. 

To solve this, the researchers used the natural properties of the weights. Since the weights were tending toward 0 anyway, if you add then to the identity function, the output will tend toward the identity.

Consider the output of a CNN layer with input $x$ and output $H(x)$
$$H(x)=x+f(x)$$
Where $f(x)$ is the change in the input to get the output. The researchers hypothesized that if a model is able to get $H(x)$ from $x$, then it should also be able to get $f(x)$ from $x$. So they set up the convolutional layers to output $f(x)$, a tensor the same shape as $x$. Since the weights tend to 0, we know that $f(x)$ will also tend to 0. By adding the identity function, if $f(x)$ tends to 0, it'll instead tend to the identity ($x$).

Original Output: $f(x)$
Residual Output: $f(x)+x$

Residual layers simply add the input of a *block* (multiple feed-forward layers) to the output of the block. This makes the deeper layers tend toward the identity meaning they make smaller changes to the output.