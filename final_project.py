import matplotlib.pyplot as plt
import numpy as np
import os,glob

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from PIL import Image

categories = ["food", "interior", "exterior"]
batch_size = 10
img_height = 300
img_width = 300
channel_num = 3
num_categories = len(categories)
epoch = 3

def plot_loss_curve(history):
    plt.figure(figsize = (15,10))
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc = 'upper right')
    
def classification_multi_preformace_eval(y,y_predict):
    tp_a,tp_b,tp_c = 0,0,0
    b_a, c_a = 0,0
    a_b, c_b = 0,0
    a_c, b_c = 0,0
    
    for y, yp in zip(y,y_predict):
        if y == 0 and yp == 0:
            tp_a += 1
        elif y == 1 and yp == 1:
            tp_b += 1
        elif y == 2 and yp == 2:
            tp_c += 1
        elif y == 0 and yp == 1:
            a_b +=1
        elif y == 0 and yp == 2:
            a_c +=1
        elif y == 1 and yp == 0:
            b_a +=1
        elif y == 1 and yp == 2:
            b_c +=1
        elif y == 2 and yp == 0:
            c_a +=1
        else:
            c_b += 1
            
    accuracy = (tp_a+tp_b+tp_c)/(tp_a+tp_b+tp_c+a_b+a_c+b_a+b_c+c_a+c_b)
    if((tp_a+b_a+c_a) == 0):
        precision_a = 0
    else:
        precision_a = (tp_a)/(tp_a+b_a+c_a)
    if((tp_a+a_b+a_c) == 0):
        recall_a = 0
    else:
        recall_a = (tp_a)/(tp_a+a_b+a_c)
    if((precision_a+recall_a == 0)):
        f1_score_a = 0
    else:
        f1_score_a = 2*precision_a*recall_a /(precision_a+recall_a)
        
    
    if((tp_b+a_b+c_b) == 0):
        precision_b = 0
    else:
        precision_b = (tp_b)/(tp_b+a_b+c_b)
    if((tp_b+b_a+b_c) == 0):
        recall_b = 0
    else:
        recall_b = (tp_b)/(tp_b+b_a+b_c)
    if((precision_b+recall_b == 0)):
        f1_score_b = 0
    else:
        f1_score_b = 2*precision_b*recall_b /(precision_b+recall_b)
    
    if((tp_c+a_c+b_c) == 0):
        precision_c = 0
    else:
        precision_c = (tp_c)/(tp_c+a_c+b_c)
    if((tp_c+c_a+c_b) == 0):
        recall_c = 0
    else:
        recall_c = (tp_c)/(tp_c+c_a+c_b)
    if((precision_c+recall_c == 0)):
        f1_score_c = 0
    else:
        f1_score_c = 2*precision_c*recall_c /(precision_c+recall_c)
        
    return accuracy,precision_a,recall_a,f1_score_a,precision_b,recall_b,f1_score_b,precision_c,recall_c,f1_score_c  

def get_img_dataset():
    PATH = os.getcwd()
    path_dir = PATH + '\\images\\'   
    
    X = []
    y = []
    
    for index, cat in enumerate(categories):
        label = [0 for i in range(num_categories)]
        label[index] = 1
    
        image_dir = path_dir + cat
        files = glob.glob(image_dir+"/*.jpg")
        print(cat, " 파일 길이 : ", len(files))
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            data = np.asarray(img)

            X.append(data)
            y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

    np.save("./numpy_train/X_train.npy",X_train)
    np.save("./numpy_train/y_train.npy",y_train)
    np.save("./numpy_test/X_test.npy",X_test)
    np.save("./numpy_test/y_test.npy",y_test)
    
def train_clusting_model():
    X_train = np.load('./numpy_train/X_train.npy')
    y_train = np.load('./numpy_train/y_train.npy')
    X_test = np.load("./numpy_test/X_test.npy")
    y_test = np.load("./numpy_test/y_test.npy")
    
    model = Sequential([Input(shape=(300,300,3), name= 'input_layer'),
                        Conv2D(32, (3,3), padding="same", activation='relu'),
                        MaxPooling2D(pool_size=(2,2)),
                        Dropout(0.25),
                        Flatten(),
                        Dense(num_categories, activation = 'softmax', name = 'output_layer', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev =0.05, seed =42))])
    model.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs=epoch, validation_data=(X_test, y_test))
    
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss = ",history.history['loss'][-1])
    print("valtrain loss = ",history.history['val_loss'][-1])
    
    model.save('model-201611189')
    print("모듈 정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))

    return model

def predict_image(model):
    X_test = np.load("./numpy_test/X_test.npy")
    y_test = np.load("./numpy_test/y_test.npy")
    y_predict = model.predict(X_test)
    y_test_label = []
    y_predict_label = []
    for i in y_test:
        pre_ans = i.argmax()
        y_test_label.append(pre_ans)
        
    for i in y_predict:
        pre_ans = i.argmax()
        y_predict_label.append(pre_ans)
    
    acc, prec_food, rec_food, f1_food, prec_interior, rec_interior, f1_interior, prec_exterior, rec_exterior, f1_exterior = classification_multi_preformace_eval(y_test_label, y_predict_label)  
        
    print("전체 accuracy = %f" %acc)
    print("=========================")
    print("food precision = %f" %prec_food)
    print("food recall = %f" %rec_food)
    print("food f1_score = %f\n" %f1_food)
    print("interior precision = %f" %prec_interior)
    print("interior recall = %f" %rec_interior)
    print("interior f1_score = %f\n" %f1_interior)
    print("exterior precision = %f" %prec_exterior)
    print("exterior recall = %f" %rec_exterior)
    print("exterior f1_score = %f\n" %f1_exterior)
    
if __name__ == '__main__':
    #get_img_dataset()
    #model = train_clusting_model()
    model = load_model('model-201611189')
    predict_image(model)