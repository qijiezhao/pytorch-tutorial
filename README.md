###数据类型
pytorch中有两种变量类型，一个是Tensor，一个是Variable。 

- Tensor： 就像ndarray一样,一维Tensor叫Vector，二维Tensor叫Matrix，三维及以上称为Tensor 
- Variable：是Tensor的一个wrapper，不仅保存了值，而且保存了这个值的creator，和他的梯度。 需要BP的网络都是Variable参与运算。 两个重要属性requires_grad和volatie表示是否需要求梯度、是否放弃存储的历史信息。

torch.autograd.Variable, 

![](http://i.imgur.com/T8hZF5Q.png)

前行计算完成后，只需要执行.backward()就能自动求得梯度。

-----------

###基本操作
torch对于数组/矩阵的操作和numpy非常相似

比如有如下表达：
#
	a=torch.Tensor(4)#生成一个长度为4的vector
	a=torch.rand(3,4)    b=torch.t/transpose(a)为转置，a[:2],a[1:2,2:3]
	a=torch.ones/zeros(3,4)，torch.nonzero([...])
	a=torch.add(torch.Tensor(3),torch.Tensor(3))
	a=torch.eye(4)
	a=torch.range(start,end,step)
	a=torch.cat(seq,dim=0) 类似于numpy.concatenate(seq,dim=0) #按axis合并矩阵/向量
	a=torch.tan(torch.Tensor(3))
	a.view(2,6) #变形
	a=torch.from_numpy(ndarry)#还可以从numpy格式转来
	...
	http://pytorch.org/docs/torch.html

----------

一些简单的基本操作：

【Input】
#
    import torch
    x  = torch.Tensor(2,3,4) #torch.Tensor(shape) 创建出一个未初始化的Tensor,但是还是可以打印出值的，
							 这个应该是这块内存之前的数据
    # x = torch.rand(shape) 生成size=shape的正随机数， x = torch.randn(shape) 生成size=shape的有符号随机数
	print x 
【Output】
#
    (0 ,.,.) = 
    1.00000e-37 *
       1.5926  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
    
    (1 ,.,.) = 
    1.00000e-37 *
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
       0.0000  0.0000  0.0000  0.0000
    [torch.FloatTensor of size 2x3x4]
【Input】
#
	x.size()
【Output】
#
	torch.Size([2, 3, 4])
【Input】
#
	a.add_(b) # 所有带 _ 的operation，都会更改调用对象的值，
				例如 a=1;b=2; a.add_(b); a就是3了，没有 _ 的operation就没有这种效果，只会返回运算结果
	torch.cuda.is_available()
【Output】
#
	True

###Variable自动求梯度的一些性质

【Input】
#
	import torch
	from torch.autograd import Variable
	w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)#需要求导的话，requires_grad=True属性是必须的。
	w2 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
	print(w1.grad)
	print(w2.grad)
	
	#w1,w2的梯度都为0

	d = torch.mean(w1)
	d.backward()
	w1.grad
	
	#w1,w2的梯度都为1/3

	d.backward()
	w1.grad
	
	#w1,w2的梯度都为2/3

	w1.grad.data.zero_()
	d.backward()
	w1.grad

	#正确的操作


.creator可以找到创造这个变量的变量和函数

【Input】
#
	x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
	y = autograd.Variable(torch.Tensor([4., 5., 6]), requires_grad=True)
	z = x + y
	print z.creator
【Output】

#
	<torch.autograd._functions.basic_ops.Add object at 0x7f2163a509e8>
利用梯度更新参数，手动做减法
#
	learning_rate = 0.01
	for f in net.parameters():
        #f.data-=f.grad.data * learning_rate
    	f.data.sub_(f.grad.data * learning_rate)  # 有了optimizer就不用写这些了

pytorch提供了很多优化算法来提供自动的参数更新

#
	import torch.optim as optim
	# create your optimizer
	optimizer = optim.SGD(net.parameters(), lr = 0.01)
	
	# in your training loop:
	for i in range(steps):
	    optimizer.zero_grad() # zero the gradient buffers，必须要置零
	    output = net(input) # 得到网络的输出结果
	    loss = criterion(output, target) # loss函数
	    loss.backward()
	    optimizer.step() # Does the update

putorch求梯度、更新参数的过程非常灵活，如下是对部分数据进行操作

【Input】
#
	import torch
	import torch.cuda as cuda
	from torch.autograd import Variable
	w1 = Variable(cuda.FloatTensor(2,3), requires_grad=True)
	res = torch.mean(w1[1])# 只用了variable的第二行参数
	res.backward()
	print(w1.grad)
【Output】
#
	Variable containing:
	 0.0000  0.0000  0.0000
	 0.3333  0.3333  0.3333
	[torch.cuda.FloatTensor of size 2x3 (GPU 0)]

----------

###整体NN结构

#
	import torch.nn as nn # layers like conv,fc
	import torch.nn.functional as F # operations like relu,sigmoid,pool

	class Net(nn.Module): # nn.Module是所有神经网络的基类，我们定义的神经网络需要继承他。
	    def __init__(self):
	        super(Net, self).__init__()
	        #建立了两个卷积层，self.conv1, self.conv2，注意，这些层都是不包含激活函数的
	        self.conv1 = nn.Conv2d(1, 6, 5) # (1 input image channel, 6 output channels, 5x5 square convolution kernel)
	        self.conv2 = nn.Conv2d(6, 16, 5)
	        #三个全连接层
	        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
	        self.fc2   = nn.Linear(120, 84) #  (input_size,  output_size)
	        self.fc3   = nn.Linear(84, 10) 
	
	    def forward(self, x): #注意，2D卷积层的输入data维数是 batchsize*channel*height*width
	        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
	        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
	        x = x.view(-1, self.num_flat_features(x))
	        x = F.relu(self.fc1(x))
	        x = F.relu(self.fc2(x))
	        x = self.fc3(x)
	        return x
	
	    def num_flat_features(self, x):# 得到数据展开后的总长度
	        size = x.size()[1:] # all dimensions except the batch dimension
	        num_features = 1
	        for s in size:
	            num_features *= s
	        return num_features

	net = Net()

	#create your optimizer
	optimizer = optim.SGD(net.parameters(), lr = 0.01)

	# in your training loop:
	for i in range(num_iteations): #训练的迭代次数
	    optimizer.zero_grad() # zero the gradient buffers，如果不归0的话，gradients会累加
	
	    output = net(input) # 得到网络的输出结果，
	
	    loss = criterion(output, target)
	    loss.backward() # 得到grad，i.e.给Variable.grad赋值
	    optimizer.step() # Does the update，i.e. Variable.data -= learning_rate*Variable.grad





所有的Tensor、Variable数据都可以转化成cuda版本：

#
	# 将Tensor放到Cuda上
	if torch.cuda.is_available():
	    x = x.cuda()
	    y = y.cuda()
1. 自己定义Variable的时候，记得Variable(Tensor, requires_grad = True),这样才会被求梯度，否则，是不会求梯度的
2. volatile和requires_grad都可以用于freeze前面的参数，volatile更灵活

【Input】
#
	import torch
	from torch.autograd import Variable
	x = Variable(torch.randn(5, 5))
	y = Variable(torch.randn(5, 5))
	z = Variable(torch.randn(5, 5), requires_grad=True)
	a = x + y  
	a.requires_grad
	b = x + z 
	b.requires_grad

【Output】
#
	False # x, y的 requires_grad的标记都为false，输出结果为关系or的结果，所以输出的变量requires_grad也为false
	True  # x, z的 requires_grad的标记分别为false和True，输出结果为关系or的结果，所有结果为True

freeze resnet18前面的conv层，只训练fc层
#
	model = torchvision.models.resnet18(pretrained=True)
	for param in model.parameters():
	    param.requires_grad = False
	# Replace the last fully-connected layer
	# Parameters of newly constructed modules have requires_grad=True by default
	model.fc = nn.Linear(512, 100)
	
	# Optimize only the classifier
	optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

###volatile
在纯推断模式的时候，只要是输入volatile=True，那么输出Variable的volatile必为True。这就比使用requires_grad=False方便多了

【Input】
#
	j = Variable(torch.randn(5,5), volatile=True)
	k = Variable(torch.randn(5,5))
	m = Variable(torch.randn(5,5))
	n = k+m # k,m变量的volatile标记都为False，输出的Variable的volatile标记也为false
	n.volatile
【Output】
#
	False

【Input】
#
	o = j+k #k,m变量的volatile标记有一个True，输出的Variable的volatile为True
	o.volatile
【Output】

#
	True

----------

###training data
pytorch团队帮我们写了一个torchvision包。使用torchvision就可以轻松实现数据的加载和预处理

#
	import torchvision # torchvision.datasets内部提供了常用的经典数据集，比如MNIST，CIFAR10，ImageNet-12，COCO等
	import torchvision.transforms as transforms #用于数据预处理

#
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=None)
	# root: path of the training/testing data.
	# train: True=Training data, False=Testing data
	# download: True=download the data online, False=load the data offline.
	# transform: pre-process the data.

split the dataset to mini-batches.
#
	import torch
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2)
	# batch_size: how many images in each mini_batch.
	# shuffle: True= shuffle the data before splitted to mini_batches.
	# num_workers: how many subprocesses to use for data loading, default 0=data will be load in the main process.

pre-process

#
	transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
								  tranforms.Scale(...),transforms.Centercrop/Ranomcrop(...)
	])
	#ToTensor: transform the value from (0,255) to (0,1), type from PIL.Image or Numpy to Tensor 
	#Normalize(mean,std):channel=(channel-mean)/std ,then value is (-1,1)

in the training process
#
	#迭代开始，然后，队列和线程跟着也开始
	data_iter = iter(trainloader)
	
	# mini-batch of input images and labels
	images, labels = data_iter.next()
	
	for images, labels in train_loader:
	    # training code (TBA)
	    pass

###Save and load models
#
	# 保存和加载整个模型
	torch.save(model_object, 'model.pkl')
	model = torch.load('model.pkl')

#
	# 仅保存和加载模型参数(推荐使用)
	torch.save(model_object.state_dict(), 'params.pkl')
	model_object.load_state_dict(torch.load('params.pkl'))

###自定义自己的datasets
#
	class CustomDataset(data.Dataset):#需要继承data.Dataset
	    def __init__(self):
	        # TODO
	        # 1. Initialize file path or list of file names.
	        pass
	    def __getitem__(self, index):
	        # TODO
	        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
	        # 2. Preprocess the data (e.g. torchvision.Transform).
	        # 3. Return a data pair (e.g. image and label).
	        #这里需要注意的是，第一步：read one data，是一个data
	        pass
	    def __len__(self):
	        # You should change 0 to the total size of your dataset.
	        return 0
	#具体请参考MNIST加载数据集的代码：[anaconda_root]/lib/python2.7/site-packages/torchvision/datasets/mnist.py

###hook
forward:
#
	import torch
	from torch import nn
	import torch.functional as F
	from torch.autograd import Variable
	
	def for_hook(module, input, output):
	    print(module)
	    for val in input:
	        print("input val:",val)
	    for out_val in output:
	        print("output val:", out_val)
	class Model(nn.Module):
	    def __init__(self):
	        super(Model, self).__init__()
	    def forward(self, x):
	        return x+1
	model = Model()
	x = Variable(torch.FloatTensor([1]), requires_grad=True)
	handle = model.register_forward_hook(for_hook)
	print(model(x))
	handle.remove()
backward:

【Input】
#
	import torch
	from torch import nn
	import torch.functional as F
	from torch.autograd import Variable
	
	def back_hook(module, grad_in, grad_out):
	    print("hello")
	    print(grad_in)
	    print(grad_out)
	
	class Model(nn.Module):
	    def __init__(self):
	        super(Model, self).__init__()
	    def forward(self, x):
	        return torch.mean(x+1)
	
	model = Model()
	x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad=True)
	#handle = model.register_forward_hook(for_hook)
	model.register_backward_hook(back_hook)
	res = model(x)
	res.backward()
【Output】
#
	hello
	(Variable containing:
	 0.3333
	 0.3333
	 0.3333
	[torch.FloatTensor of size 3],)
	(Variable containing:
	 1
	[torch.FloatTensor of size 1],)

----------

定义一个RNN：
![](http://i.imgur.com/xGBJD9W.png)
#
	import torch.nn as nn
	from torch.autograd import Variable
	class RNN(nn.Module):
	    def __init__(self, input_size, hidden_size, output_size):
	        super(RNN, self).__init__()
	        self.hidden_size = hidden_size
	        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
	        self.i2o = nn.Linear(input_size + hidden_size, output_size)
	        self.softmax = nn.LogSoftmax()
	    def forward(self, input, hidden):
	        combined = torch.cat((input, hidden), 1)
	        hidden = self.i2h(combined)
	        output = self.i2o(combined)
	        output = self.softmax(output)
	        return output, hidden
	    def initHidden(self):
	        return Variable(torch.zeros(1, self.hidden_size))
	n_hidden = 128
	rnn = RNN(n_letters, n_hidden, n_categories)

训练网络：
#
	learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
	def train(category_tensor, line_tensor):
	    hidden = rnn.initHidden()
	    rnn.zero_grad()
	    for i in range(line_tensor.size()[0]):
	        output, hidden = rnn(line_tensor[i], hidden)
	    loss = criterion(output, category_tensor)
	    loss.backward()
	# update the parameters
	    for p in rnn.parameters():
	        p.data.add_(-learning_rate, p.grad.data)
	return output, loss.data[0]
