from image_test.nn.conv.lenet import LeNet
from image_test.nn.conv.minivggnet import MiniVGGNet
from keras.utils import plot_model


model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="./output/lenet.png", show_shapes=True)

model2 = MiniVGGNet.build(32, 32, 1, 10)
plot_model(model2, to_file="./output/minivggnet.png", show_shapes=True)
