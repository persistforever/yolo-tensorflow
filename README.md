# tensorflow版本的YOLO



## 模型介绍

### 1. 目标函数

#### anyobject loss

anyobject loss用以约束所有应该预测出没有物体的预测框(pred box)的5维信息，即这些预测框对应的5维向量(x, y, w, h, c)应该接近于(0, 0, 0, 0, 0)。反映在目标函数中，表示为，
$$
\sum_{x=1}^M \sum_{y=1}^M \sum_{j=1}^B \bf{1}_{xyj}^{noobj} (c_{xyj} - \hat{c}_{xyj})^2 , c_{xyj}=0
$$
反映在伪代码中，

```python
anyobject_mask = numpy.zeros((n_cell, n_cell, n_boxes, 1), dtype=float)
for x in range(n_cell):
  for y in range(n_cell):
    for j in range(n_boxes):
      pred_box = pred_boxes[x,y,j,0:4]
      max_iou = 0
      for k in range(max_objects):
        true_box = true_boxes[k,0:4]
        iou = calculate_iou(pred_box, true_box)
        if iou >= max_iou:
          max_iou = iou
      if max_iou > anyobject_thresh:
        anyobject_mask[x,y,j,0] = 1.0
anyobject_true = numpy.zeros((n_cell, n_cell, n_boxes, 1), dtype=float)
anyobject_pred = predict[:,:,:,4:]
anyobject_loss = l2_loss((anyobject_true - anyobject_pred) * anyobject_mask)
```



#### coord loss, object loss, class loss

coord loss用以约束所有应该预测出物体的预测框的前4维坐标信息，即这些预测框对应的4维坐标向量(xp, yp, wp, hp)应该接近于真实的物体坐标信息(xt, yt, wt, ht)。反映在目标函数中，表示为，
$$
\sum_{x=1}^M \sum_{y=1}^M \sum_{j=1}^B \bf{1}_{xyj}^{obj} \left[ (x_{xyj} - \hat{x}_{xyj})^2 + (y_{xyj} - \hat{y}_{xyj})^2 + (\sqrt{w}_{xyj} - \sqrt{\hat{w}}_{xyj})^2 + (\sqrt{h}_{xyj} - \sqrt{\hat{h}}_{xyj})^2 \right]
$$
object loss用以约束所有应该预测出物体的预测框的第5维置信度，即这些预测框对应的置信度cp应该接近于1。反映在目标函数中，表示为，
$$
\sum_{x=1}^M \sum_{y=1}^M \sum_{j=1}^B \bf{1}_{xyj}^{obj} (c_{xyj} - \hat{c}_{xyj})^2 , c_{xyj}=1
$$
class loss用以约束所有应该预测出物体的预测框的第6到N+5维类别信息，若真实的类别为第n类，那么这个预测框对应的第n维应该接近于1，第6到N+5维中其余的维度应该接近于0。反映在目标函数中，表示为，
$$
\sum_{x=1}^M \sum_{y=1}^M \sum_{j=1}^B \bf{1}_{xyj}^{obj} (p_{xyj} - \hat{p}_{xyj})^2
$$
指的注意的是，类别为n的真实框所覆盖的所有cell对应的类别信息，都应该是第n维为1，其余为0，而非只是真实框的中心点覆盖的cell。反映在伪代码中，

```python
object_mask = numpy.zeros((n_cell, n_cell, n_boxes, 1), dtype=float)
coord_loss, object_loss, class_loss = 0.0, 0.0, 0.0
for k in range(max_objects):
  true_box = true_boxes[k,0:4]
  cell_x, cell_y = int(true_box[0]), int(true_box[1])
  max_iou, max_index = 0, 0
  for j in range(n_boxes)
  	pred_box = pred_boxes[cell_x,cell_y,j,0:4]
    iou = calculate_iou(pred_box, true_box)
    if iou >= max_iou:
      max_iou = iou
      max_index = j
  object_mask[cell_x,cell_y,j,0] = 1.0
  
  coord_true = numpy.tile(true_boxes[k,0:4], [n_cell,n_cell,n_boxes,4])
  coord_pred = predict[:,:,:,0:4]
  coord_loss += l2_loss((coord_true - coord_pred) * object_mask)
  
  object_true = numpy.ones((n_cell,n_cell,n_boxes,1), dtype=float)
  object_pred = predict[:,:,:,4:5]
  object_loss += l2_loss((object_true - object_pred) * object_mask)
  
  class_true = numpy.zeros((n_cell,n_cell,n_boxes,n_class), dtype=float)
  class_true[cell_x,cell_y,:,4+true_boxes[k,5]] = 1.0
  class_pred = predict[:,:,:,5:]
  class_loss += l2_loss((class_true - class_pred) * object_mask)
```

