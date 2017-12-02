from multiprocessing import Process, Queue
import time
import numpy
import random
import cv2

n_iter = 40
n_processes = 4
batch_size = 32

def consume(data):
    time.sleep(0.5)

def produce():
    st = time.time()
    w, h = random.randint(500, 600), random.randint(800,900)
    images = numpy.array(numpy.random.random((batch_size, w, h, 3)), dtype='float32')
    new_images = []
    for j in range(batch_size):
        new_images.append(cv2.resize(images[j], dsize=(448, 448)))
    new_images = numpy.array(new_images, dtype='float32')
    labels = numpy.array(numpy.random.random((batch_size, 100)), dtype='int32')
    data = [new_images, labels]
    return data

def main():
    avg_time = 0.0
    for i in range(n_iter):
        st = time.time()
        data = produce()
        et = time.time()
        print('get data time: %.4f' % (et -st))
        avg_time += et - st
        consume(data)
    print('average get data time: %.4f' % (1.0 * avg_time / n_iter))

if __name__ == '__main__':
    main()
