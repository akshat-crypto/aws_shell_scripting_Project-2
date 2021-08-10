import tensorflow as tf
from tensorflow import keras
import tensorflow.python.keras.utils.generic_utils
from tensorflow.keras.applications import VGG16
#from keras import VGG16
model = VGG16(#weights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                         include_top = False, 
                         input_shape = (224, 224, 3))
model.summary()

for layer in model.layers:
        layer.trainable = False

layer.__class__


# In[6]:


layer.__class__.__name__


# In[7]:


layer.__class__.__name__, layer.trainable


# In[8]:


enumerate(model.layers)

for (i,layer) in enumerate(model.layers):
        print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


        # In[10]:


        def LayerAddflatten(bottom_model, num_classes):
                """creates the top or head of the model that will be 
                    placed ontop of the bottom layers"""
                top_model = bottom_model.output
                top_model = Flatten(name = "flatten")(top_model)
                top_model = Dense(526, activation = "relu")(top_model)
                top_model = Dense(263, activation = "relu")(top_model)
                top_model = Dense(num_classes, activation = "sigmoid")(top_model)
                return top_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


num_classes = 2

FC_Head = LayerAddflatten(model, num_classes)

modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 16
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
        ##change this path accd to the requirement first
        '/home/ec2-user/shell_scripting_aws_project2/ml/dataset/training_set',
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        ##change this path accd to the requirement first
        '/home/ec2-user/shell_scripting_aws_project2/ml/dataset/test_set',
        target_size=(224, 224 ),
        batch_size=val_batchsize,
        class_mode='categorical')

from tensorflow.keras.optimizers import RMSprop

modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop( lr = 0.001 ),
              metrics = ['accuracy'])

nb_train_samples = 1190
nb_validation_samples = 170
#epochs = 3
batch_size = 20
history = modelnew.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size,
    epochs = 1)

result_accuracy = history.history['accuracy']


print ("model is trained and accuracy is:" , result_accuracy)
templist1=list(result_accuracy)
print(templist1)
tempmax=max(templist1)
op=int(tempmax * 100)
print()
print("MAX ACCURACY IS: ",op)
print()
modelnew.save('dog_cat_transfer-learning.h5')
