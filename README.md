#SSD Face Detection
An implemention of SSD for face detection. The database used in this project is FDDB. There are around 3000 images with annotations in FDDB.
Consider the images has different sizes, I append black area to each image in different ways to resize the image. I saved the original size 
and 300X300 size to increase the data set. The final dataset used for train has around 18000 images. The reason I select SSD is when I did 
reseach at the beging of this project, I noticed that YOLO and Faster-RCNN have been used for face detection and there are not a lot of projects
about SSD in Github.
#Current Status
I used VGG 16 for SSD. I only trained the model for 10 epoches. Also, I uploaded the trained model, if you want to train more epoches, you can use 
the model directly. 
#Future Work
I will try to implement the SSD based on MobileNet. According some papers, SSD could have different speed with various models. The MobileNet has the 
fasted speed, though it has a lower accurancy. Later in this month, I will rewrite some codes and upload the dataset pre process part. The next step of 
this project is add a face landmark detection.
