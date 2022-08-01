# ResNet
```
Paper   || [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Authors || Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
```

An implementation of the original ResNet architecture as outlined in the 2015 paper, Deep Residual Learning for Image Recognition. This pytorch implementation achieves 92.13% test accuracy for resnet20 and 93.52% test accuracy for resnet50 on the CIFAR-10 dataset—within 3% of the author's reported results.

GRAPH

## Implementation Details
### Shortcut Connections
The heart of the ResNet architecture is the residual connections (aka shortcut connections) that combine the feed-forward layer output with the input (identity). This way, deeper layers that would normally detract from the models performance instead have a baseline of not changing it.
$$\sigma=\text{activation function}$$
$$\text{output from previous layer}=x$$
$$\text{output from this layer}=\sigma(f(x))$$
$$\text{output after residual connection}=\sigma(f(x)+x)$$
If the weights of the layer tend to 0, then f(x) will also tend to 0. Normally this would harm model performance, but by adding the input to the output, it instead trends toward the input, thus not changing the model’s performance.
```python
def forward(self, input: torch.Tensor):
	# assume block_layers is some set of feed-forward layers
	output = self.block_layers(input)
	
	# shortcut connection
	output += input
	return self.activation(output)
```
