r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1.The shape of the Jacobian tensor of the output of the layer w.r.t. the input X would be:
(N, out_features, in_features) which is (128, 2048, 1024)."

2.The number of elements in the Jacobian tensor is N * out_features * in_features = 128 * 2048 * 1024.
Since each element is represented using single-precision floating point (32 bits),
the total memory required to store the Jacobian tensor is 128 * 2048 * 1024 * 4 bytes= 1073741824 bytes.
In gigabytes, this is approximately 1 GB of RAM or GPU memory."""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.1
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1e-3
    lr_vanilla = 2e-2
    lr_momentum = 2e-3
    lr_rmsprop = 1e-4
    reg = 1e-3
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-3
    lr = 3e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

Yes, the graphs matched what I expected to see. When the dropout is 0, we have the highest training accuracy because 
the model can perfectly learn or even overfit to the training data. But this leads to poor generalization to the 
unseen test data, and thus the lowest test accuracy.

When the dropout is 0.4, the training accuracy decreases slightly, suggesting that the model is not fitting the 
training data as perfectly as before. This is expected, as dropout randomly turns off neurons during training, 
making the model less sensitive to specific features and more robust. However, this very feature of dropout helps in 
increasing the test accuracy, as it prevents overfitting, and thus the model is better able to generalize to unseen 
data.

Finally, with a dropout rate of 0.8, the model starts to underfit the training data, as can be seen by the further 
decrease in training accuracy. This happens because a high dropout rate is turning off a large portion of neurons 
during training, thereby limiting the model's capacity to learn from the data. However, interestingly, 
the test accuracy for dropout 0.8 is still higher than for dropout 0. This could be because, despite underfitting, 
the model generalizes better than the overfitted model (dropout 0). However, it's important to note that the test 
accuracy for dropout 0.8 is still lower than for dropout 0.4, showing that there is an optimal balance to strike when 
setting the dropout rate to prevent both overfitting and underfitting. Too little or too much dropout can hinder 
performance on the test set."""

part2_q2 = r"""**Your answer:** Yes, it is indeed possible for the test loss to increase for a few epochs while the 
test accuracy also increases during the training of a machine learning model. This might seem counterintuitive at 
first glance, but it can be explained by the different ways that loss and accuracy measure the performance of a model.

The cross-entropy loss measures the distance between the model's predicted probabilities and the true labels. It 
penalizes confident but wrong predictions harshly and rewards correct predictions, especially confident ones. Loss 
values, thus, are very sensitive to the confidence of the predictions.

On the other hand, accuracy is a discrete metric that merely counts the proportion of correctly classified instances. 
It doesn't consider the confidence of the predictions. For example, a prediction with 51% confidence and a prediction 
with 99% confidence are treated the same in terms of accuracy as long as they are both correct.

Let's consider a scenario where initially the model is making less confident but correct predictions. This will lead 
to lower accuracy but also lower loss because the cross-entropy loss is relatively low for less confident 
predictions. As the model learns more from the training data and becomes more confident in its predictions, 
the accuracy might increase (as it might start to get more predictions correct), but the loss might also increase if 
any of these confident predictions are wrong because cross-entropy loss heavily penalizes confident and incorrect 
predictions.

In summary, the loss (in this case, cross-entropy loss) is sensitive to the confidence of the predictions while 
accuracy is not. This discrepancy can lead to situations where accuracy increases while the loss also increases."""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
1. Number of Parameters:

In a regular block of two 3x3 convolutions, each with 256 input channels and 256 output channels, the total number of 
parameters would be:

For each convolution layer: (kernel height * kernel width * input channels * output channels) = (3 * 3 * 256 * 256) = 
589,824 parameters. Since there are two such layers in a block, the total parameters = 2 * 589,824 = 1,179,
648 parameters.

In a bottleneck block, the structure typically includes a 1x1 convolution that reduces the dimensionality, 
then a 3x3 convolution, followed by another 1x1 convolution that restores the original depth. Let's assume we're 
reducing to 64 channels, as is common in ResNet-50. The number of parameters would be:

For the first 1x1 convolution: (1 * 1 * 256 * 64) = 16,384 parameters.

For the middle 3x3 convolution: (3 * 3 * 64 * 64) = 36,864 parameters.

For the final 1x1 convolution: (1 * 1 * 64 * 256) = 16,384 parameters.

Adding these up, the total parameters in a bottleneck block would be: 16,384 + 36,864 + 16,384 = 69,632 parameters.

2. Number of Floating Point Operations (FLOPs):

Qualitatively, the bottleneck block is more computationally efficient due to the dimensionality reduction in the 
middle layer. The most computationally heavy operation (3x3 convolution) happens on the reduced dimension (64 
channels instead of 256 channels), which saves on the number of FLOPs.

3. Ability to Combine the Input:

Both blocks are capable of combining inputs spatially within feature maps due to their 3x3 convolution operations. 

As for combining across feature maps, both types of blocks can do this. However, the bottleneck block does this more 
extensively because of the two 1x1 convolutions, which serve as a kind of "channel shuffling". The 1x1 convolutions 
allow the network to learn to construct new feature maps by combining different input maps, potentially enabling it 
to extract more useful features.
"""

part3_q2 = r"""**Your answer:** it appears that increasing the depth of the network (L) beyond a certain point (in 
this case, beyond L=4) decreases the model's performance significantly. The accuracy dropping to around 10% for L=8 
and L=16 suggests that the models with these configurations are struggling to learn from the data effectively. This 
is a likely indication of the vanishing gradient problem, which tends to become more severe as the network depth 
increases. This problem causes the gradients of the loss function to become very small, making the weights of the 
initial layers hard to train.

The reason the network with L=2 has higher accuracy than L=4 could be due to overfitting. As the network becomes 
deeper, it has more parameters and thus more capacity to fit to the training data. If this capacity isn't controlled 
properly (e.g., with regularization), it can lead to overfitting, where the model learns the training data too well 
and performs poorly on unseen data.

Use Residual Connections: Also known as skip connections, they were introduced in ResNet (Residual Network) to 
alleviate the vanishing gradient problem. They do this by providing a shortcut for the gradient to flow through. If 
you add these connections to your network, the gradients can bypass layers, which can prevent them from becoming too 
small.

Batch Normalization: Batch normalization can also help mitigate the vanishing/exploding gradients problem. It works 
by normalizing the output of each layer to maintain a mean close to 0 and standard deviation close to 1. This ensures 
that no values become excessively large or excessively small, thus maintaining stable gradients."""

part3_q3 = r"""
**Your answer:**

 The best performing models had a depth of L=2 or L=4, with L=4 achieving the best results. Models with depth L=8 and 
 beyond struggled to learn, resulting in poor performance, which is consistent across both experiments. This likely 
 results from the vanishing/exploding gradients problem, which tends to become more severe as network depth increases.

In Experiment 1.2, varying the number of filters per layer showed that K=128 gave the best results when 
the depth was L=4. This suggests that increasing the width of the model (i.e., the number of filters per layer) can 
improve performance up to a point, as it allows the model to learn more diverse features. However, extremely wide 
models may suffer from overfitting if not managed properly.

Comparing results from both experiments, the key takeaway seems to be that a balance of depth and width is important 
for optimal performance. Too much depth can lead to issues with gradient vanishing/exploding, making the network 
difficult to train. On the other hand, increasing width can improve performance by increasing the capacity of the 
model, but care should be taken to prevent overfitting.

For the depth of L=4, the model seemed to be in a "sweet spot" where it was deep enough to learn complex features but 
not so deep as to encounter severe issues with vanishing/exploding gradients. The width of K=128 provided enough 
capacity for the model to learn diverse features without overfitting, resulting in the best performance.

"""

part3_q4 = r"""**Your answer:** 
It seems that a single layer (L=1) model with K=[64,128,256] performs significantly 
better (75% accuracy) compared to models with higher depths (L=2,3,4), which all hovered around 10% accuracy.

When comparing these results with the previous experiments, the general theme seems to be that deeper networks (L=4 
or more) consistently underperform compared to shallower ones. This could be an indication that the task or the data 
doesn't benefit from deeper architectures, or that other techniques are needed to effectively train deeper models (
e.g., regularization, different optimization algorithms, etc.).
"""

part3_q5 = r"""**Your answer:**
 
Residual Connections Impact: Residual connections have improved the performance of 
deeper networks. For instance, test accuracies for configurations L=8, K=32 and L=16, K=32 are now around 70%, 
compared to 10% in previous experiments without residual connections.

Depth and Width Influence: With K=[64, 128, 256], the best test performance was seen with L=2 (74% accuracy), 
declining as depth increased. Wider networks (K=[64, 128, 256]) performed better than narrower ones (K=32).

Overfitting: Small gaps between training and test accuracy suggest mild overfitting. Overfitting-prone models might 
benefit from regularization, dropout, or larger training sets.

Trainability: All network configurations were trainable, likely due to residual connections. However, performance 
significantly dropped with L=32, K=32 (40% training and 42% test accuracy).

These results highlight the utility of techniques like residual connections in training deeper networks, and the need 
for a balance between network depth and width for optimal performance. Increasing depth or width isn't always 
beneficial.
"""

part3_q6 = r"""**Your answer:** 
1. **Architecture Modifications**: In our custom network (`YourCodeNet` class), 
we added Dropout (0.4) and Batch Normalization to address limitations from Experiment 1.

2. **Results Analysis**: Experiment 2 achieved 74% accuracy for `L=3`, outperforming Experiment 1. The additions of 
Dropout and Batch Normalization improved model performance and robustness."""
# ==============
