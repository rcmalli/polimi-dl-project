from src.model import unpool_resize,unpool_deconv, unpool_checkerboard, unpool_simple
from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.models import Model

input = Input(shape=(20, 20, 3))

out1 = unpool_resize(input)
model1 = Model(inputs=input, outputs=out1)
print("")

out2 = unpool_deconv(input,512)
model2 = Model(inputs=input, outputs=out2)
print("")

out3 = UpSampling2D((2,2))(input)
out3 = unpool_checkerboard(out3)
model3 = Model(inputs=input, outputs=out3)
print("")

