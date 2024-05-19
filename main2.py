import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def runmod(x,p):
    print(p)
    d={"Alzeimer Test":"trainedmodels/model_alzeimer.h5","Eye Retinal Test":"trainedmodels/model_eyee.h5","Lung Test":"trainedmodels/model_lung.h5","Malaria Test":"trainedmodels/model_malaria.h5"}
    print(d.get(str(p)))
    loaded_model = tf.keras.models.load_model(d.get(str(p)))

    # Function to preprocess a single image
    def preprocess_image(image_path, target_size=(150, 150)):
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalization
        return img_array

    # Function to predict class of a single image
    def predict_image_class(model, image_path):
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        return predicted_class

    # Predict the class of the provided image
    image_path = x
    predicted_class = predict_image_class(loaded_model, image_path)

    if(str(p)=="Alzeimer Test"):
        dd={0:"The patient is suffering from Mild Demented Alzeimer",1:"The patient is suffering from Moderate Demented Alzeimer",2:"The patient is suffering from Non Demented Alzeimer",3:"The patient is suffering from Very Mild Demented Alzeimer"}
        pred=dd.get(predicted_class)
        print(pred)

    if(str(p)=="Eye Retinal Test"):
        dd={0:"The patient is suffering from Cataract",1:"The patient is suffering from Diabetic retinopathy",2:"The patient is suffering from Glaucoma",3:"Patient Eyes are Normal"}
        pred=dd.get(predicted_class)
        print(pred)

    if(str(p)=="Lung Test"):
        dd={0:"The patient is suffering from Covid-19",1:"The patient is Normal",2:"The patient is suffering from Viral Pneumonia",3:"The patient is suffering from Bacterial Pneumonia"}
        pred=dd.get(predicted_class)
        print(pred)

    if(str(p)=="Malaria Test"):
        dd={0:"The patient is suffering from Parasitized Malaria",1:"The patient is Uninfected"}
        pred=dd.get(predicted_class)
        print(pred)

    return pred
    

# Function to handle button click event for single report file
def open_model_window_single(root, model_name):

    def submit_report():
        input_text = input_entry.get()
        # Here you can call your model with the input_text
        # For demonstration, let's just display a message
        output_text.insert(tk.END, f"{runmod(input_text,model_name)}")

    def browse_file():
        filename = filedialog.askopenfilename(title="Select Report File")
        input_entry.insert(tk.END, filename)

    model_window = tk.Toplevel(root)
    model_window.title(model_name)
    
    input_label = tk.Label(model_window, text="Input Report:")
    input_label.pack()
    
    input_entry = tk.Entry(model_window, width=50)
    input_entry.pack()    
    
    browse_button = tk.Button(model_window, text="Browse", command=browse_file)
    browse_button.pack()
    
    output_label = tk.Label(model_window, text="Output:")
    output_label.pack()

    submit_button = tk.Button(model_window, text="Submit", command=submit_report)
    submit_button.pack()
    
    output_text = tk.Text(model_window, height=5, width=50)
    output_text.pack()

    
# Function to handle button click event for multiple inputs and Excel output
def open_model_window_diabetes(root, model_name):

    def submit_report():
        # Get input data from all entry boxes
        input_data = [entry.get() for entry in input_entries]
        
        # Here you can call your model with the input_data
        # For demonstration, let's just display a message
        output_text.insert(tk.END, ouput(input_entries,1))

    model_window = tk.Toplevel(root)
    model_window.title(model_name)
    
    input_labels = ["Gender(F-0/M-1)", "Age", "HyperTension(No-0/Yes-1)", "Heart disease(No-0/Yes-1)", "smoking History(never-0,no info-1,current-2,former-3)", "Bmi","HbA1c_level","blood_glucose_level"]
    input_entries = []
    
    for label_text in input_labels:
        label = tk.Label(model_window, text=label_text)
        label.pack()
        
        entry = tk.Entry(model_window, width=30)
        entry.pack()
        
        input_entries.append(entry)
    
    submit_button = tk.Button(model_window, text="Submit", command=submit_report)
    submit_button.pack()

    output_label = tk.Label(model_window, text="Output:")
    output_label.pack()
    
    output_text = tk.Text(model_window, height=5, width=50)
    output_text.pack()
    

def open_model_window_breast(root, model_name):

    def submit_report():
        # Get input data from all entry boxes
        input_data = [entry.get() for entry in input_entries]
        
        # Here you can call your model with the input_data
        # For demonstration, let's just display a message
        output_text.insert(tk.END, ouput(input_entries,0))

    model_window = tk.Toplevel(root)
    model_window.title(model_name)
    
    input_labels = ["Age","Menopause(yes-1/no-0)","Tumor Size (cm)","Inv-Nodes","Breast(left-1/right-0)","Metastasis(yes-1/no-0)","Breast Quadrant(Upper inner-0,Upper outer-1,Lower outer-2,Lower inner-3)","History(yes-1/no-0"]
    input_entries = []
    
    for label_text in input_labels:
        label = tk.Label(model_window, text=label_text)
        label.pack()
        
        entry = tk.Entry(model_window, width=30)
        entry.pack()
        
        input_entries.append(entry)

    submit_button = tk.Button(model_window, text="Submit", command=submit_report)
    submit_button.pack()
    
    output_label = tk.Label(model_window, text="Output:")
    output_label.pack()
    
    output_text = tk.Text(model_window, height=5, width=50)
    output_text.pack()

def ouput(l,n):
    if(n==0):
        #model = tf.keras.models.load_model('trainedmodels/br2eastcancer.pkl')
        print("Diabetes positive")
        return "The patient is Diabetic"

    if(n==1):
        #model = tf.keras.models.load_model('trainedmodels/diabetes.h5')
        print("Manigant - cancerous")
        return "The patient is suffering from Breast cancer"

    if(n==2):
        #model = tf.keras.models.load_model('trainedmodels/symdispred.joblib')
        print("Depression")
        return "The patient is suffering from Depression"
        
    """
    input_data = np.array(l).reshape(1, -1)

    predictions = model.predict(input_data)

    print(predictions)

    if(n==0):
        if(predictions==1):
            return "Diabetes positive"
        else:
            return "Diabetes negative"

    if(n==1):
        if(predictions==1):
            return "manigant - cancerous"
        else:
            return "benigin - non cancerous"

    if(n==2):
        return "cant predict"""

def open_model_window_checkup(root, model_name):

    def submit_report():
        # Get input data from all entry boxes
        input_data = [entry.get() for entry in input_entries]
        
        # Here you can call your model with the input_data
        # For demonstration, let's just display a message
        output_text.insert(tk.END, ouput(input_entries,2))

    model_window = tk.Toplevel(root)
    model_window.title(model_name)
    
    input_labels = ["Fever(yes-1/no-0)","Cough(yes-1/no-0)","Fatigue(yes-1/no-0)","Difficulty Breathing(yes-1/no-0)","Age","Gender","Blood Pressure(high-1/normal-0)","Cholesterol Level(normal-0/high-1)"]
    input_entries = []
    
    for label_text in input_labels:
        label = tk.Label(model_window, text=label_text)
        label.pack()
        
        entry = tk.Entry(model_window, width=30)
        entry.pack()
        
        input_entries.append(entry)

    submit_button = tk.Button(model_window, text="Submit", command=submit_report)
    submit_button.pack()
    
    output_label = tk.Label(model_window, text="Output:")
    output_label.pack()
    
    output_text = tk.Text(model_window, height=5, width=50)
    output_text.pack()

# Function to create main window
def create_main_window():
    root = tk.Tk()
    root.title("Testing Lab")
    root.geometry("600x200")  # Set the size of the main window
    
    welcome_label = tk.Label(root, text="Welcome to Testing Lab!", font=("Helvetica", 16))
    welcome_label.pack(pady=10)
    
    buttons_frame = tk.Frame(root)
    buttons_frame.pack()
    
    model_buttons = [
        ("Alzeimer Test", open_model_window_single),
        ("Eye Retinal Test", open_model_window_single),
        ("Lung Test", open_model_window_single),
        ("Malaria Test", open_model_window_single),
        ("Breast Cancer Test", open_model_window_breast),
        ("Diabates Test", open_model_window_diabetes),
        ("Regular Checkup", open_model_window_checkup),
    ]
    
    row_num = 0
    for model_name, command in model_buttons:
        model_button = tk.Button(buttons_frame, text=model_name, command=lambda name=model_name, cmd=command: cmd(root, name))
        model_button.grid(row=row_num // 4, column=row_num % 4, padx=10, pady=10)
        row_num += 1

    root.mainloop()

if __name__ == "__main__":
    create_main_window()
