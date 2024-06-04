import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import numpy as np

def detect_barcode(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use pyzbar to detect barcodes in the image
    barcodes = decode(gray)

    # Loop over the detected barcodes
    for barcode in barcodes:
        # Extract the barcode data and type
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Draw a rectangle around the barcode on the image
        points = barcode.polygon
        if len(points) == 4:
            pts = np.array([(point.x, point.y) for point in points], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Print the barcode data and type
        print(f"Barcode Type: {barcode_type}, Data: {barcode_data}")

    # Display the image with detected barcodes using Matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Detected Barcodes')
    plt.axis('off')
    plt.show()

# Example usage
image_path = r"""C:\Users\ivsva\OneDrive\Desktop\360_F_455480661_B1ndlageM3kplzg1NRPFUgYj2iWXvDQS.jpg"""
detect_barcode(image_path)