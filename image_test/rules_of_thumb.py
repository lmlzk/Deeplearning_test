# the input layer should be square

# the input layer should be divisible by two multiple times after the first
# CONV operation is applied

# use smaller filter sizes such as 3*3\5*5
# 1*1 are used to learn local features in more advanced network
# larger filter sizes such as 7*7\11*11 may be used as the first CONV layer

# commonly use a stride of S=1 for CONV layers unless skipped max pooling
# networks accept larger input volumes that use stride S>=2 in the first CONV layer

# apply zero-padding

# use POOL layers reduce the spatial dimensions of the input

# pooling applied over a 2*2/3*3 receptive field size and a stride of S=2
# it is highly uncommon to see receptive fields larger than three since these
# operations are very destructive to their inputs

# while BN does indeed slow down the training time, it also tends to "stabilize" training,
# making it easier to tune other hyperparameters

# place the batch normalization after the activation

# Dropout is typically applied in between FC layers with a dropout p of 50%
# while not always performed, yoou should consider applying DO in nearly every arc

#