from multiprocessing import Process, Queue
import time
import numpy
import random
import cv2

n_iter = 40
n_processes = 4
batch_size = 64

# consumer module: using trainable data to train model
def consume(queue):
    time.sleep(30)
    avg_time = 0.0
    for j in range(n_iter):
        st = time.time()
        res = queue.get()
        et = time.time()
        avg_time += et - st
        print('get data time: %.4f' % (et - st))
        time.sleep(0.5)
    print('average get data time: %.4f' % (1.0 * avg_time / n_iter))

# producer module: using origin data to generate trainable data
def produce(queue):
    for i in range(n_iter / n_processes):
        st = time.time()
        w, h = random.randint(500, 600), random.randint(800,900)
        images = numpy.array(numpy.random.random((batch_size, w, h, 3)), dtype='float32')
        new_images = []
        for j in range(batch_size):
            new_images.append(cv2.resize(images[j], dsize=(448, 448)))
        new_images = numpy.array(new_images, dtype='float32')
        labels = numpy.array(numpy.random.random((batch_size, 100)), dtype='int32')
        queue.put([new_images, labels])
        et = time.time()
        print('produce data time: %.4f' % (et - st))

def main():
    queue = Queue(maxsize=10)

    producer_list = []
    for i in range(n_processes):
        producer = Process(target=produce, args=(queue,))
        producer_list.append(producer)
    consumer = Process(target=consume, args=(queue,))
    
    for producer in producer_list:
        producer.start()
    consumer.start()


if __name__ == '__main__':
    main()
