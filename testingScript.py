import numpy as np
import csv


def generateTestingCSV(model, X_test, test_images_names):
    predictions = model.predict(X_test)
    with open('result.csv', 'w' , newline='') as csv_file:
        fieldnames = ['image_name', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(predictions)):
            writer.writerow({'image_name': test_images_names[i], 'label': np.argmax(predictions[i])})

