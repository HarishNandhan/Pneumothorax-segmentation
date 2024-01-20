import os
import cv2
import imgaug.augmenters as iaa

# Define the augmentation techniques you want to apply
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # flip horizontally with a probability of 0.5
    iaa.GaussianBlur(sigma=(0, 3.0)),  # add Gaussian blur with sigma between 0 and 3.0
    iaa.Affine(rotate=(-45, 45))  # rotate between -45 and 45 degrees
])

# Define the input folder
input_folder = r'D:\College\Semester 7\Project\Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow\Data\No Pnuemothorax'

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    img = cv2.imread(os.path.join(input_folder, filename))

    # Apply the augmentation sequence to the image
    augmented_img = seq(image=img)

    # Save the augmented image to the same folder with a modified filename
    cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)

# Define the input folder
input_folder = r'D:\College\Semester 7\Project\Pneumothorax-Binary-Classification--OpenCV--Keras--Tensorflow\Data\Pnuemothorax'

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    # Load the image
    img = cv2.imread(os.path.join(input_folder, filename))

    # Apply the augmentation sequence to the image
    augmented_img = seq(image=img)

    # Save the augmented image to the same folder with a modified filename
    cv2.imwrite(os.path.join(input_folder, 'aug_' + filename), augmented_img)
