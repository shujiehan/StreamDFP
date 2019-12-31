# aggregate instances with same serial number in a evaluated window


class Instances:
    def __init__(self, sn, queue_size):
        self.sn = sn
        self.queue = []
        self.queue_size = queue_size

    def enqueue(self, inst):
        assert (len(self.queue) <= self.queue_size)
        self.queue.append(inst)

    def dequeue(self):
        del self.queue[0]
