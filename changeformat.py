import tensorflow as tf
print("starting")
# Load your model from the .pkl file
model = tf.keras.models.load_model('trainedmodels/diabetespred.pkl')

# Save the model to the H5 format
model.save('trainedmodels/diabetespred.h5')
print("succesful")