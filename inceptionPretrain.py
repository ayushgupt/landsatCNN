from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import data_prepr
from keras import metrics
from keras.optimizers import Adam

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator( data_prepr.get_batch_data(), epochs=2,steps_per_epoch=500) 

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
#   layer.trainable = False
for layer in model.layers[:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=[ metrics.categorical_accuracy])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator( data_prepr.get_batch_data(),epochs=2, steps_per_epoch=500) 
test_batch_size=64
total_test_batch= 500
avg=0

for i in range(total_test_batch):
	evalX,evalY=data_prepr.get_eval_data()
	loss,acc = model.evaluate(evalX,evalY, batch_size=64)
	print("loss: ",loss)
	print("acc: ",acc)
	avg+=acc
print("avg: ", avg/total_test_batch)
	# avg+=eval_results["accuracy"]
# print("test_accuracy: ",avg/total_test_batch)

# total_train_batch=train_len//test_batch_size
# avg=0
