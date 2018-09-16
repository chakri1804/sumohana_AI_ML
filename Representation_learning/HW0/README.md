### Please run all the codes in Python 3
### In case of K-Means and PCA, different images have been included for testing purposes. In case a code takes more time, lower resolution images have also been attached for quicker computation

### Failure case for PCA:

Consider a ellipse drawn in R2 centred at origin (focii on equal spacing from origin) and some noise added to say 10,000 equally spaced coordinates on that ellipse

In this case, applying PCA will project the ellipse entirely onto X axis or Y axis. But the integrity of the ellipse itself is lost after PCA.

![](/home/legion/Documents/sem_5/sumohana_AI_ML/Representation_learning/HW0/Screenshot from 2018-09-16 19-29-14.png)

Instead, applying PCA on local patches (like a kernel striding in a CNN) may result in the underlying ellipse. 
