# -*- coding: utf-8 -*-
# author: ronniecao
# time: 2018/01/11
# description: util tools for project
import os


def is_in_table(text, table):
    """
    判断文本框text是否在table中
    输入1：text - 文本框，[left, top, right, bottom]
    输入2：table - 表格框，[left, top, right, bottom]
    """
    cross_left = max(int(text[0]), int(table[0]))
    cross_top = max(int(text[1]), int(table[1]))
    cross_right = min(int(text[2]), int(table[2]))
    cross_bottom = min(int(text[3]), int(table[3]))
    if cross_left <= cross_right and cross_top <= cross_bottom:
        return True
    else:
        return False
    
def cal_resized_size(orig_h, orig_w, canvas_h, canvas_w):
    """
    将尺寸为[orig_h, orig_w]的图像进行等比例缩放后的尺寸[resized_h, resized_w]
    """
    if 1.0 * orig_h / orig_w < 1.0 * canvas_h / canvas_w:
        resized_h = int(1.0 * canvas_w / orig_w * orig_h)
        resized_w = int(canvas_w)
        is_horizal = True
    else:
        resized_h = int(canvas_h)
        resized_w = int(1.0 * canvas_h / orig_h * orig_w)
        is_horizal = False

    return resized_h, resized_w, is_horizal
