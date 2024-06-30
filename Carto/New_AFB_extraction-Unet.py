#!/usr/bin/env python
# coding: utf-8

# In[45]:


import tensorflow as tf
from segmentation_models import Unet, Linknet, PSPNet, FPN
import segmentation_models as sm
from keras.utils import plot_model
import albumentations as A
import numpy as np
from numpy import savez_compressed,savez
from numpy import load
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score  


# ## Trained and tested using only TN data

# In[3]:


data = load('E:/carto_1_256_final.npz')


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(data['arr_0'], data['arr_1'], test_size=0.10, random_state=123)


# In[5]:


print(X_train.shape)
print(X_test.shape)


# In[6]:


# Vertical Flip
transform1 = A.Compose([A.VerticalFlip(p=1)])
vflip_images = []
vflip_labels = []
for i in range(0,len(X_train)):
    transformed1 = transform1(image=X_train[i], mask=y_train[i])
    vflip_images.append(transformed1['image'])
    vflip_labels.append(transformed1['mask']) 

#%% Horizontal Flip
transform2 = A.Compose([A.HorizontalFlip(p=1)])
hflip_images = []
hflip_labels = []
for i in range(0,len(X_train)):
    transformed2 = transform2(image=X_train[i], mask=y_train[i])
    hflip_images.append(transformed2['image'])
    hflip_labels.append(transformed2['mask'])

#%% Transpose
transform3 = A.Compose([A.Transpose(p=1)])
tp_images = []
tp_labels = []
for i in range(0,len(X_train)):
    transformed3 = transform3(image=X_train[i], mask=y_train[i])
    tp_images.append(transformed3['image'])
    tp_labels.append(transformed3['mask']) 

#%% #ShiftScaleRotate 
transform4 = A.Compose([A.ShiftScaleRotate(p=1)])
ssr_images = []
ssr_labels = []
for i in range(0,len(X_train)):
    transformed4 = transform4(image=X_train[i], mask=y_train[i])
    ssr_images.append(transformed4['image'])
    ssr_labels.append(transformed4['mask'])

#%% #RandomFog 
transform5 = A.Compose([A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.2, always_apply=True, p=1)])
fog_images = []
fog_labels = []
for i in range(0,len(X_train)):
    transformed5 = transform5(image=X_train[i], mask=y_train[i])
    fog_images.append(transformed5['image'])
    fog_labels.append(transformed5['mask'])
    
#%% #Rotate90 
transform6 = A.Compose([A.SafeRotate(limit=90, p=1)])
t90_images = []
t90_labels = []
for i in range(0,len(X_train)):
    transformed6 = transform6(image=X_train[i], mask=y_train[i])
    t90_images.append(transformed6['image'])
    t90_labels.append(transformed6['mask'])
    
#%% #Rotate180 
transform7 = A.Compose([A.SafeRotate(limit=180, p=1)])
t180_images = []
t180_labels = []
for i in range(0,len(X_train)):
    transformed7 = transform7(image=X_train[i], mask=y_train[i])
    t180_images.append(transformed7['image'])
    t180_labels.append(transformed7['mask'])

#%% #Rotate=270 
transform8 = A.Compose([A.SafeRotate(limit=270, p=1)])
t270_images = []
t270_labels = []
for i in range(0,len(X_train)):
    transformed8 = transform8(image=X_train[i], mask=y_train[i])
    t270_images.append(transformed8['image'])
    t270_labels.append(transformed8['mask'])

#%% #Rotate360
transform9 = A.Compose([A.SafeRotate(limit=360, p=1)])
t360_images = []
t360_labels = []
for i in range(0,len(X_train)):
    transformed9 = transform9(image=X_train[i], mask=y_train[i])
    t360_images.append(transformed9['image'])
    t360_labels.append(transformed9['mask'])


# In[7]:


X_train_final = np.array(list(X_train) + hflip_images + vflip_images + tp_images + ssr_images + fog_images + t90_images + t180_images + t270_images + t360_images)
y_train_final = np.array(list(y_train) + hflip_labels + vflip_labels + tp_labels + ssr_labels + fog_labels + t90_labels + t180_labels + t270_labels + t360_labels)


# ## Custom Callbacks

# In[14]:


def lr_scheduler(epoch, lr):
    decay_rate = .95
    decay_step = 5
    if (epoch+1) % decay_step == 0 :
        return lr * decay_rate
    return lr

reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.9, patience=2, min_lr=0.000001)
learning_rate = LearningRateScheduler(lr_scheduler, verbose=1)

checkpoint_filepath = 'weights-{epoch:02d}-{accuracy:.4f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='accuracy', verbose=1,save_best_only=True, mode='max')

cb_list = [reduce_lr,learning_rate,checkpoint]


# In[15]:


model1 = Unet('vgg16',input_shape=(256, 256, 3),classes=1, activation='sigmoid',encoder_weights=None)
opt = tf.keras.optimizers.Adam(lr=0.00001)
model1.compile(optimizer=opt, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])


# In[17]:


results = model1.fit(X_train_final, y_train_final, validation_split=0.2, epochs=200,batch_size=16)


# In[18]:


def plot_performance(plot_title):
    plt.suptitle(plot_title)
    plt.subplot(1,2,1)
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.legend(['Training','Validation'],loc='upper right')
    
    plt.subplot(1,2,2)
    plt.plot(results.history['iou_score'])
    plt.plot(results.history['val_iou_score'])
    plt.title('Model IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend(['Training','Validation'],loc='lower right')
    plt.show()
    
plot_performance('Performance')


# ## Plotting test data

# In[19]:


predict = model1.predict(X_test)

def plot_predictions(num):
    plt.suptitle('Cartosat 2 and Predicted Boundaries ' + str(num))
    plt.subplot(1,3,1)
    plt.imshow(X_test[num])
    plt.title('Cartosat-2 image')

    plt.subplot(1,3,2)
    plt.imshow(predict[num],cmap="gray")
    plt.title('Predicted Boundaries')
    
    plt.subplot(1,3,3)
    plt.imshow(y_test[num],cmap="gray")
    plt.title('Actual Boundaries')
    plt.show()


# In[20]:


plot_predictions(18)


# In[30]:


plot_predictions(40)


# ## Plotting train data

# In[22]:


train_predict = model1.predict(X_train)

def plot_train_predictions(num):
    plt.suptitle('Cartosat 2 and Predicted Boundaries ' + str(num))
    plt.subplot(1,3,1)
    plt.imshow(X_train[num])
    plt.title('Cartosat-2 image')

    plt.subplot(1,3,2)
    plt.imshow(train_predict[num],cmap="gray")
    plt.title('Predicted Boundaries')
    
    plt.subplot(1,3,3)
    plt.imshow(y_train[num],cmap="gray")
    plt.title('Actual Boundaries')
    plt.show()


# In[24]:


plot_train_predictions(12)


# In[25]:


plot_train_predictions(120)


# ## Computes IoU, F1 score and Accuracy

# In[43]:


def compute_metrics(y_pred, y_true,x):
    # ytrue, ypred is a flatten vector
    y_pred = np.round(y_pred[x].flatten())
    y_true = np.round(y_true[x].flatten())
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    f1 = f1_score(y_true,y_pred)
    acc = accuracy_score(y_true,y_pred)
    return np.mean(IoU), f1, acc


# In[49]:


iou = []
f1score = []
acc = []

for i in range(len(y_test)):
    a,b,c = compute_metrics(predict,y_test,i)
    iou.append(a)
    f1score.append(b)
    acc.append(c)


# In[52]:


print(np.mean(iou))
print(np.mean(f1score))
print(np.mean(acc))


# In[47]:


for i in range(len(y_train)):
    print(compute_metrics(train_predict,y_train,i))


# In[ ]:




