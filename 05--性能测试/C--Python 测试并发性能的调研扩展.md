# C--Python 测试并发性能的调研扩展



<span style='color:brown'>**本文探究的核心问题：**</span>

1. Python可以同时运行的线程数量及制约因素；
2. 单位时间内Python可以启动的线程数量及制约因素；



**参考链接：**

1. [The right way to limit maximun number of threads running at once ?](https://stackoverflow.com/questions/19369724/the-right-way-to-limit-maximum-number-of-threads-running-at-once)
2. [How many Python threads can i run ?](https://www.quora.com/How-many-Python-threads-can-I-run)
3. [Queue -- A synchronized queue class](https://docs.python.org/3.8/library/queue.html)



## 1、The right way to limit maximun number of threads running at once?

原文地址：

- [The right way to limit maximun number of threads running at once ?](https://stackoverflow.com/questions/19369724/the-right-way-to-limit-maximum-number-of-threads-running-at-once)

核心诉求：

- I'd like to create a program that runs multiple light threads, but limits itself to a constant, predefined number of concurrent running tasks, like this (but with no risk of race condition)

  > 我想创建一个程序，运行多个轻量级线程，但将自己限制在一个恒定的、预定义的并发运行任务数量上，就像这样（但没有竞赛条件的风险）。

<span style='color:brown'>**Answer -- 1 :**</span>

听起来你想用八个工作者来实现生产者/消费者模式。Python 有一个 [Queue 类](https://docs.python.org/2/library/queue.html)用于这个目的，而且它是线程安全的。

每个工作人员都应该在队列上调用 get() 来检索任务。如果没有可用的任务，此调用将阻塞，导致工作人员空闲，直到有一个可用。然后工作人员应该执行任务，最后在队列上调用 task_done()。

你可以通过在队列上调用 put() 将任务放入队列中。

在主线程中，你可以在队列上调用 join() 以等待所有待处理的任务完成。

这种方法的好处是你不会创建和销毁线程，这很昂贵。工作线程将连续运行，但当队列中没有任务时将进入睡眠状态，使用零 CPU 时间。

>  队列不直接限制线程，但它允许您通过使用池轻松限制线程，链接示例准确显示了如何做到这一点。并且在队列中存储更多数据根本不会降低系统性能，或者至少不会超过将其存储在列表中；它只是一个双端队列周围的一些锁，它占用的存储空间不超过一个列表。



<span style='color:brown'>**Answer -- 2 :**</span>

我遇到了同样的问题，并花了几天（准确地说是 2 天）使用队列找到正确的解决方案。我在 ThreadPoolExecutor 路径上浪费了一天，因为没有办法限制事物启动的线程数！我给它提供了一个包含 5000 个要复制的文件的列表，一旦它同时运行了大约 1500 个并发文件副本，代码就会无响应。 ThreadPoolExecutor 上的 max_workers 参数仅控制有多少工作线程正在启动线程，而不是有多少线程启动。

这是一个非常简单的使用队列的例子：

```python
import threading, time, random
from queue import Queue

jobs = Queue()

def do_something(q):
    while not q.empty():
        value = q.get()
        time.sleep(random.randint(1, 10))
        print(value)
        q.task_done()
        
for i in range(10):
    jobs.put(i)
    
for i in rnage(3):
    worker = threading.Thread(target=do_something, args=(jobs, ))
    worker.start()
    
print("waiting for queue to complete", jobs.qsize(), "tasks")
jobs.join()
print('all done')
```

补充说明：

>  ThreadPoolExecutor上的max_workers参数只控制了有多少工作者在旋转线程，而不是有多少线程被旋转起来。如果你把它设置为1，那么你就会得到单线程的性能。如果你把它设置为2，而你的队列中有几千个长期运行的任务，那么这两个工作者就会开始旋转线程，直到他们为每个项目都旋转了一个线程才会停止。如果这些任务在争夺相同的资源，如内存、存储或网络，你就会有一个大问题。





## 2、How many Python threads can I run ?

原文地址：

- [How many Python threads can i run ?](https://www.quora.com/How-many-Python-threads-can-I-run)

<span style='color:brown'>**Answer -- 1 :**</span>

只有一个！

好吧，这并不完全正确。事实是，你可以在 Python 中运行尽可能多的线程，只要你有内存，但 Python 进程中的所有线程都运行在一个机器核心上，所以从技术上讲，实际上一次只有一个线程在执行。

这意味着 Python 线程实际上只对并发 I/O 操作有用。它们无法加速（实际上可以减慢）CPU 密集型任务。

我的建议是将 asyncio 用于 I/O 绑定并发并在进程中运行 CPU 密集型任务。进程池通常是最简单的方法。 concurrent.futures.ProcessPoolExecutor 与 ascyncio 兼容。

线程主要用于 Python 中的小规模、I/O 绑定并发，在这种情况下，你不希望处理异步代码的认知开销带来麻烦 (并不是说它非常困难，而是线程更容易)。



## 3、If a thread is a task in CPU, how can the CPU run 150 more than processes ?

原文地址：

- [How can the CPU run 150 more than processes ?](https://qr.ae/pviivO)

单个 CPU 内核可以支持一个或多个并发执行的硬件线程。单个内核可以支持的硬件线程数量没有理论上的限制。大多数处理器内核只支持一个硬件线程。现代台式机和工作站 x86 处理器内核支持每个内核两个硬件线程。

多任务操作系统管理任意数量的软件线程。操作系统调度程序负责获取当前有工作要做的线程，并将它们分配给处理器内核上的硬件线程。

在任何时候，硬件只能并行地执行与硬件线程一样多的活动软件线程。操作系统负责在可用的硬件线程上对软件线程进行时间复用。

<span style='color:brown'>**应该指出，并发性和并行性不是一回事：**</span>

- 并发线程意味着线程的生命期是重叠的：
  - 如果其中一个线程在另一个线程完成其生命周期之前开始其生命周期，则线程 A 与线程 B 是并发的。
  - 并发线程不需要以同样的速度取得进展。事实上，这是并发的一个优势：你可以将一个阻塞的任务卸载到后台线程，继续在其他工作上取得进展。
- 并行线程意味着线程实际上是在并发地执行每个线程的指令：
  - 例如，如果你有两个硬件核心，线程A在核心0上运行的同时，线程B在核心1上运行，那么这两个线程是并行运行的。
  - 执行指令……并发意味着硬件主动为线程发出指令，以便来自不同线程的各个指令的生命周期可能重叠。这包括在多个内核上执行的线程，以及单个内核上的多个活动硬件线程。

并行线程始终是并发的，而并发线程并不总是并行执行。

<span style='color:brown'>**你的 $150$ 或 $100,000$ 或任何软件线程并发执行。你的机器拥有的硬件线程的数量决定了有多少个并发线程可以并行执行。**</span>



## 4、Queue -- A synchronized queue class

**官方文档：**

- [queue -- A synchronized queue class](https://docs.python.org/3.8/library/queue.html)



queue 模块实现了多生产者、多消费者的队列。当必须在多个线程之间安全地交换信息时，它在线程编程中特别有用。该模块中的 Queue 类实现了所有必需的锁定语义。

该模块实现了三种类型的队列，它们只在检索条目的顺序上有所不同。在先进先出（FIFO）队列中，第一个添加的任务是第一个被检索的。在后进先出队列中，最近添加的条目是第一个被检索的（操作类似于堆栈）。在优先级队列中，条目被保持排序（使用 [**heapq**](https://docs.python.org/3.8/library/heapq.html#module-heapq) 模块），价值最低的条目被首先检索。

在内部，这三种类型的队列使用锁来暂时阻断竞争的线程；但是，它们并没有被设计用来处理线程内的重入。

此外，该模块还实现了一个 "简单 "的先进先出队列类型，即SimpleQueue，其具体实现提供了额外的保证，以换取较小的功能。

队列模块定义了以下类和异常：

- class queue.Queue(maxsize=0)

  FIFO队列的构造函数。 maxsize是一个整数，用于设置可以放在队列中的项目数量的上限。一旦达到这个大小，插入将被阻止，直到队列项目被消耗。如果maxsize小于或等于0，队列的大小就是无限的。

- class queue.LifoQueue(maxsize=0)

  后进先出队列的构造函数。 maxsize是一个整数，它设置了可以放在队列中的项目数量的上限值。一旦达到这个大小，插入将被阻止，直到队列项目被消耗。如果maxsize小于或等于0，则队列的大小是无限的。

- class queue.PriorityQueue(maxsize=0)

  一个优先级队列的构造函数。 maxsize是一个整数，它设置了可以放在队列中的项目数量的上限值。一旦达到这个大小，插入将被阻止，直到队列项目被消耗。如果maxsize小于或等于0，队列的大小就是无限的。

  首先检索价值最低的条目(最低值的条目是由 sorted(list(entries))[0]返回的条目)，条目的典型模式是以下形式的元祖：(priority_number, data)。

  如果数据元素没有可比性，可以将数据包裹在一个类中，忽略数据项，只比较优先级比：

  ```python
  from dataclasses import dataclass, field
  from typing import Any
  
  @dataclass(order=True)
  calss PrioritizedItem:
      priority: int
          item: Any=field(compare=False)
  ```

- class queue.SimpleQueue

  无界 FIFO 队列的构造函数。简单队列缺少任务跟踪等高级功能。

- exception queue.Empty

  当对一个空的队列对象调用非阻塞的 get() (或 get_nowait())时，会产生异常。

- exception queue.Full

  当在一个已满的队列对象上调用非阻塞的 put() (或put_nowait())时，会产生异常。



### Queue Objects

队列对象（Queue、LifoQueue或PriorityQueue）提供下面描述的公共方法。

- Queue.qsize()

  返回队列的近似大小。注意，qsize() > 0并不能保证随后的get()不会阻塞，qsize() < maxsize也不能保证put()不会阻塞。

- Queue.empty()

  如果队列是空的，返回True，否则返回False。如果empty()返回True，它并不保证随后对put()的调用不会阻塞。同样，如果empty()返回False，也不能保证随后对get()的调用不会阻塞。

- Queue.full()

  如果队列已满，返回True，否则返回False。如果full()返回True，并不能保证随后对get()的调用不会阻塞。同样，如果full()返回False，也不能保证随后对put()的调用不会阻塞。

- Queue.put(item, block=True, timeout=None)

  将项目放入队列。如果可选的args block为true，并且timeout为None（默认值），必要时进行阻塞，直到有空闲的槽。如果timeout是一个正数，它最多阻断timeout秒，如果在这段时间内没有空闲槽，则引发Full异常。否则（block为false），如果有空闲的槽立即可用，就在队列上放一个项目，否则就引发Full异常（在这种情况下忽略超时）。

- Queue.put_nowait(item)

  等价于 put(item, False)。

- Queue.get(block=True, timeout=None)

  从队列中移除并返回一个项目。如果optional args block为true，并且timeout为None（默认），则必要时阻塞，直到有一个项目可用。如果timeout是一个正数，它最多阻断时间，如果在该时间内没有项目可用，则引发Empty异常。否则（block为false），如果有一个项目立即可用，则返回一个项目，否则引发Empty异常（在这种情况下，超时被忽略了）。

  在POSIX系统的3.0版本之前，以及Windows系统的所有版本，如果block为真，timeout为None，这个操作就会进入对底层锁的不间断等待。这意味着不会有异常发生，特别是SIGINT不会触发键盘中断。

- Queue.get_nowait()

  Equivalent to `get(False)`.

- Queue.task_done()

  表明以前排队的任务已经完成。由队列消费者线程使用。对于每一个用于获取任务的get()，对task_done()的后续调用告诉队列该任务的处理已经完成。

  如果一个join()当前处于阻塞状态，当所有的项目都被处理完后，它将恢复（意味着每个被放入队列的项目都收到了task_done()调用）。

  如果调用次数多于队列中放置的项目，则引发ValueError。

- Queue.join()

   阻断，直到队列中的所有项目都被获取和处理。

  每当一个项目被添加到队列中，未完成的任务数量就会上升。每当消费者线程调用task_done()表示项目被检索到并且所有的工作都已完成，该计数就会下降。当未完成任务的数量下降到零时，join()就会解除阻塞。



如何等待排队任务完成的示例：

```python
import threading, queue

q = queue.Queue()

def worker():
    while True:
        item = q.get()
        print(f'Working on {item}')
        print(f'Finished {item}')
        q.task_done()
        
# turn-on the worker thread
threading.Thread(target=worker, daemon=True).start()

# send thirty task requests to the worker
for item in range(30):
    q.put(item)
print('All task requests sent\n', end='')

# block until all tasks are done
q.join()
print('All work completed')
```



### Combining Threading and Queue

```python
import queue
import threading
import time

def do_something(q, thread_no):
    while True:
        task = q.get()
        time.sleep(2)
        q.task_done()
        print(f'Thread #{thread_no} is doing task #{task} in the queue.')
        
q = queue.Queue()

for i in range(4):
    worker = threading.Thread(target=do_something, args=(q, i,), daemon=True)
    worker.start()
    
for j in range(10):
    q.put(j)
    
q.join()
```



## <span style='color:brown'>5、Threading Timer Thread in Python</span>

原文地址：

- [Threading Timer Thread in Python](https://superfastpython.com/timer-thread-in-python/)

<span style='color:brown'>第一需求：</span>

- 在并发编程中，有时我们可能希望一个线程在某个时间限制过后开始执行我们的代码。这可以用一个定时器线程来实现。

  **我们如何在Python中使用一个定时器线程呢？**



### How to Use a Timer Thread

首先，我们可以创建一个定时器的实例并对其进行配置。这包括在执行前要等待的时间（秒），一旦触发要执行的函数，以及目标函数的任何参数。

```python
# configure a timer thread
timer = Time(10, task, args=(arg1, arg2))
```

目标任务函数将不会执行，直到时间过了。

一旦创建，线程必须通过调用 start() 函数来启动，该函数将启动计时器。

```python
# start the timer thread
timer.start()
```



### Example of Using a Timer Thread

在这个例子中，我们将使用一个定时器来延迟一些处理，在这个例子中，在一个等待期后报告一个自定义的消息。

首先，我们可以定义一个目标任务函数，它接收一个消息，然后用打印语句报告。

```python
# target task function
def task(message):
    # report the custom message
    print(message)
```

接下来，我们可以创建一个threading.Timer类的实例。我们将为Timer配置3秒的延迟，然后调用task()函数，其单参数消息为 "Hello world"。

```python
# create a thread timer object
timer = Timer(3, task, args=('hello world'))
```

然后可以启动线程：

```python
# start the timer object
timer.start()
```

最后，主线程将等待定时器线程完成后退出。

```python
# wait for the timer to finish
print('Waiting for the timer...')
```

完整代码如下：

```python
# SuperFastPython.com
# example of using a thread timer object
from threading import Timer

# target task function
def task(message):
    # report the custom message
    print(message)

# create a thread timer object
timer = Timer(3, task, args=('Hello world',))
# start the timer object
timer.start()

# wait for the timer to finish
print('Waiting for the timer...')
```


