import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print (x_train.shape, y_train.shape)

#count=1
#idd=0
#plt.figure(figsize = (10,10))
#for i in range (5):
#  for j in range (2):
#    plt.subplot (5,5,count)
#    plt.imshow(x_train[idd], cmap='gray')
#    plt.axis('off')
#    plt.title('Classe: '+str(y_train[idd]))
#    idd+=1
#    count+=1
#plt.show()

model = tf.keras.applications.MobileNetV2((32, 32, 1), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


history = model.fit (x_train, y_train, epochs=25, batch_size=128, validation_split=0.2, verbose=1)


exit()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="[::]:8080", client=CifarClient())
