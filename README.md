# Convolutional-Neural-Net-Pipeline-Optimization-and-Training

Keyword: scratch... I have no doubt this is very messy code, but a notebook style approach proved very effective for getting the intial building classification model off the ground. If these training steps were to be put into production, the code should be heavily refactored.

Since the two types of structures classified (residential, aka 'house', and non-residential aka 'barn') have significant overlap in appearance in the real world, I considered a training accuracy of 79 and validation accurace of 68 to be relatively good!

Also, although a variety of regions were sampled for training data (planimetrics we had received from consultants for some of our prior project areas around the country), regional skewing should still be expected based on visible light profiles. To counter this, I added an option to normalize scores to a specific percentage of 'residential' structures. The GIS analysts user would simply enter their estimate (generally around 30-50%) and the scoring system would enforce it.
