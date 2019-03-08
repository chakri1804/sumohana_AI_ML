### Report for the CNN code

All the plots and accuracies are mentioned in the Ipython notebook attached

### Problems faced

Hard-coding differentiation functions as we did in this assignment can lead to many issues

Say one of the expressions is wrong, none of the losses converge

Tensorflow and Keras build a computational graph where all the trainable parameters can be nudged by a small value and hence partial derivative wrt any variable is computed within some margin of error. In hardcoding scenario, it's very difficult to backtrack where the issue is occuring

Also, initialisation plays a huge roll in backprop. Initialising weights as too high can lead to NaN error during backprop. So all weights are initialised with uniform distribution between 0 to 1e-3

In the following code, kernels seemed to be learning some maps but the loss and results say otherwise. Have tested this code with multiple initialisations, learning rates, shuffle seeds, no shuffles, batch GD, SGD but nothing worked and accuracy never crossed 0.1