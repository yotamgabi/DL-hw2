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
    wstd = 0.1
    lr_vanilla = 0.1
    lr_momentum = 0.01  
    lr_rmsprop = 0.001
    reg = 0.01
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
    wstd = 1e-1
    lr = 1e-3
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

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""

part3_q5 = r"""
**Your answer:**

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
