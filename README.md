# ResNet
```
Paper   || [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Authors || Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
```

An implementation of the original ResNet architecture as outlined in the 2015 paper, Deep Residual Learning for Image Recognition. This pytorch implementation achieves 92.00% test accuracy for resnet20 and 92.96% test accuracy for resnet50 on the CIFAR-10 dataset—within 1% of the author's reported results.

>This implementation is intended to be as clear and concise as possible, and is therefore not optimized for performance.

| Architecture | Test Error (Paper) | Test Error (Me)  |
| --- | --- | --- |
| ResNet20 | 8.75 % | 8.00 % |
| ResNet32 | 7.51 % | 7.43 % |
| ResNet44 | 7.17 % | 7.19 % |
| ResNet56 | 6.97 % | 7.04 % |

## Index
- [Implementation Details](https://github.com/vidithbalasa/ResNet#implementation-details)
    - [Shortcut Connections](https://github.com/vidithbalasa/ResNet#shortcut-connections)
    - [Parameters](https://github.com/vidithbalasa/ResNet#parameters)
- [Usage](https://github.com/vidithbalasa/ResNet#usage)
- [Works Cited](https://github.com/vidithbalasa/ResNet#works-cited)

## Implementation Details
### Shortcut Connections
At the heart of ResNet lie residual layers (aka shortcut connections). Weights in deeper layers have a tendency to drift to 0, leading the whole output to 0 which hinders model performance. To fix this, the authors proposed to add the input to the output so that as the weights moved to 0, the final output would move toward the input. This way layers with bad weights would simply not change model performance instead of make it worse.
$$\sigma=\text{activation function}$$
$$\text{output from previous layer}=x$$
$$\text{output from this layer}=\sigma(f(x))$$
$$\text{output after residual connection}=\sigma(f(x)+x)$$
If the weights of the layer trend towards 0, then f(x) will also trend towards 0. Normally this would harm model performance, but by adding the input to the output (x), as f(x) trends to 0 the output trends towards x.
```python
def forward(self, input: torch.Tensor) -> torch.Tensor:
	# assume block_layers is some set of feed-forward layers
	output = self.block_layers(input)
	
	# shortcut connection
	output += input
	return self.activation(output)
```
#### Special Case - Change in Size
As the model travels to each set of blocks, the authors double the number of channels to gain more info. At the same time, they add a stride of 2 to reduce the size in half, retaining the same resolution as before. This means there are instances where we need to project the model to the new output. The authors propose 2 viable methods.
>Method A
>- Downsample the image using a pool layer with stride of 2 (to match conv layer with stride of 2)
>- Double the number of channels by adding 0 value channels
>
>*avoids using any extra parameters which cuts down on train time*
```python
class Block:
	def __init__(self, ...):
		…
		self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

	def shortcut_projection(self, identity: torch.Tensor) -> torch.Tensor:
		# identity.shape           == ( 16, 32, 32 )
		# projected_identity.shape == ( 32, 16, 16 )
		
		# The feed-forward block downsamples the image by increasing stride to 2, so we do the same
		projected_identity = self.downsample(identity)
		
		# create a 0 value copy of the projected_identity
		projected_identity_zero_clone = torch.zeros_like(projected_identity)
		# add the clone to the original to double the number of channel layers
		projected_identity = torch.cat([projected_identity, projected_identity_zero_clone], dim=1)
		return projected_identity
		
```
>Method B
>- Run the input through a conv layer that doubles the channels and downsamples with stride=2 in one go
>
>*creates more accurate identity projected by not using 0 value channels*
```python
class Block:
	def __init__(self, in_channels, ...):
		# conv layer that doubles the number of channels and downsamples w stride of 2
		self.conv_project = nn.Conv2d(in_channels=in_channels, outchannels=in_channels*2, kernel_size=3, stride=2)
		# batch normalization
		self.norm = nn.BatchNorm2d(in_channels*2)
	
	def shortcut_projection(self, identity: torch.Tensor) -> torch.Tensor:
		projected_identity = self.conv_project(identity)
		return self.norm(projected_identity)
```
### Parameters
#### Training Time
Authors trained their model for 64k iterations. To match that in epochs, num_epochs should be ceil(64k / ceil(num_data / batch_size)). With a batch size of 128, this works out to 164 epochs.
#### Batch Normalization
In order to match results of the paper, batch normalization with running mean and standard deviation estimates are necessary. This can be accomplished with `nn.BatchNorm2d(…, track_running_stats=True)`
#### Hyper Parameters
I kept all the hyper parameters the same as the authors used in the paper. These can be found in `params.py`.

## Usage
### Train Locally
Open `colab_train.ipynb` and run every block after the G-Drive setup section.

### Train on Colab
Copy any paste code blocks from `colab_train.ipynb` G-Drive setup section into a colab file. Then go into the repo on drive and open `colab_train.ipynb` in colab. You can then run every block and modify it as you wish.

## Works Cited
