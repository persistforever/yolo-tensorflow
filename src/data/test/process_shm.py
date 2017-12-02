from multiprocessing import Process, Queue, Lock
from multiprocessing.sharedctypes import Array, Value
from ctypes import c_double, cast, POINTER
import time
import numpy
import random
import cv2

TYPE_PTR_FLOAT = POINTER(c_double)

buffer_size = 10
n_iter = 40
n_processes = 4
batch_size = 32
size1 = batch_size * 448 * 448 * 3
size2 = batch_size * 100
dataset_size = size1 + size2

array = Array('d', [0.0] * buffer_size * dataset_size)
_buffer = array._obj._wrapper
put_index = Value('i', 0)
get_index = Value('i', 0)
put_lock = Lock()

def consume():
    time.sleep(30)
    for i in range(n_iter):
        st = time.time()

        while put_index.value - get_index.value <= 0:
            time.sleep(0.1)
        index = get_index.value % buffer_size
        buffer_ptr = cast(_buffer.get_address() + index * dataset_size * 8, POINTER(c_double))
        data = numpy.ctypeslib.as_array(buffer_ptr, shape=(dataset_size, ))
        get_index.value += 1
        images = numpy.reshape(data[0:size1], (batch_size, 448, 448, 3))
        labels = numpy.reshape(data[size1:size1+size2], (batch_size, 100))
        
        et = time.time()
        print('get data time: %.4f' % (et - st))

        time.sleep(0.5)

def produce():
    while True:
        st = time.time()
        
        w, h = random.randint(500, 600), random.randint(800,900)
        images = numpy.array(numpy.random.random((batch_size, w, h, 3)), dtype='float32')
        new_images = []
        for j in range(batch_size):
            new_images.append(cv2.resize(images[j], dsize=(448, 448)))
        new_images = numpy.array(new_images, dtype='float32')
        labels = numpy.array(numpy.random.random((batch_size, 100)), dtype='int32')
        new_images = new_images.flatten()
        labels = labels.flatten()
        data = numpy.concatenate([new_images, labels], axis=0)
            
        with put_lock:
            while put_index.value - get_index.value >= buffer_size - 1:
                time.sleep(0.1)
            index = put_index.value % buffer_size
            _buffer_ptr = cast(_buffer.get_address() + (index * dataset_size * 8), TYPE_PTR_FLOAT)
            arr = numpy.ctypeslib.as_array(_buffer_ptr, shape=(dataset_size,))
            arr[:] = data
            put_index.value += 1
            
            et = time.time()
            print('produce data time: %.4f' % (et - st))

def main():
    producer_list = []
    for i in range(n_processes):
        producer = Process(target=produce)
        producer_list.append(producer)
    consumer = Process(target=consume)

    for producer in producer_list:
        producer.start()
    consumer.start()


if __name__ == '__main__':
    main()
