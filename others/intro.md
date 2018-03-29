## How to improve performance of object detection system

### Problem description

Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. There are several key steps in existing object detection systems, such as hypothesizing bounding boxes, resampling pixels or features for each box, and applying a high- quality classifier. Two fractions develop gradually because of different kernel goal. The two-stage detectors focus on the accuracy of object detection, which first generates a set of candidate bounding boxes and then  selects and revises target bounding boxes, such as, R-CNN (Girshick R, 2015a), Fast R-CNN (Girshick R, 2015b), Faster R-CNN (Ren, 2015), Mask R-CNN (He K, 2017), FPN (Lin T Y, 2017). In contrast, the one-state detectors focus on the efficiency of object detection, which regresses object boxes directly in an end-to-end framework, such as, SSD (Liu W, 2016), YOLO (Redmon J, 2015), YOLOv2 (Redmon J, 2016), RetinaNet (Lin T Y, 2017).



## Reference

1. Girshick R, Donahue J, Darrell T, et al. Region-Based Convolutional Networks for Accurate Object Detection and Segmentation. on PAML, 2015a.
2. Girshick, R. Fast r-cnn. in ICCV, 2015b.
3. Ren, S., He, K., Girshick, R., & Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. in NIPS, 2015.
4. He K, Gkioxari G, Dollár P, et al. Mask R-CNN. in ICCV, 2017.
5. Lin T Y, Dollar P, Girshick R, et al. Feature Pyramid Networks for Object Detection. in CVPR, 2017.
6. Liu W, Anguelov D, Erhan D, et al. SSD: Single Shot MultiBox Detector. in ECCV, 2016.
7. Redmon J, Divvala S, Girshick R, et al. You Only Look Once: Unified, Real-Time Object Detection. 2015.
8. Redmon J, Farhadi A. YOLO9000: Better, Faster, Stronger. 2016.
9. Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection. in ICCV, 2017.