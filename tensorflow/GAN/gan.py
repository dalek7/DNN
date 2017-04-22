# http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

#Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-mean',help='Mean of target gaussian',type=float)
parser.add_argument('-std',help='Standard Deviation of Target Gaussian',type=float)
parser.add_argument('-hneurons',help='Number of hidden layer neurons to use',type=int)
parser.add_argument('-epoch',help='Number of epochs to train',type=int)
parser.add_argument('-minbatch',help='Size of batch to train',type=int)
parser.add_argument('-sample',help='Size of points to sample from true distribution',type=int)
args = parser.parse_args()


#Generator and Discriminator functions. 'a_i' is pre-activation and 'h_i' is activation of ith layer
def discriminator(input, weights, bias):
	h0 = tf.to_float(input)
	a1 = tf.add(tf.matmul(h0,weights[0]),bias[0])
	h1 = tf.tanh(a1)
	a2 = tf.add(tf.matmul(h1,weights[1]),bias[1])
	y_hat = tf.sigmoid(a2)
	return y_hat

def generator(input, weights, bias):
	h0 = tf.to_float(input)
	a1 = tf.add(tf.matmul(h0,weights[0]),bias[0])
	h1 = tf.tanh(a1)
	a2 = tf.add(tf.matmul(h1,weights[1]),bias[1])
	y_hat = a2					#Not taking sigmoid because output is just real valued and not squashed
	return y_hat

#Training Parameters
EPOCHS = 20000
if(args.epoch):
	EPOCHS = args.epoch
BATCH_SIZE = 10
if(args.minbatch):
	BATCH_SIZE = args.minbatch
HLAYER_SIZE = 5
if(args.hneurons):
	HLAYER_SIZE = args.hneurons
TOTAL_SAMPLE_SIZE = 1000
if(args.sample):
	TOTAL_SAMPLE_SIZE = args.sample
TOTAL_STEPS = (TOTAL_SAMPLE_SIZE/BATCH_SIZE)*EPOCHS

mu = 5
if(args.mean):
	mu = args.mean
sigma = 1
if(args.std):
	sigma = args.std

#Define Weights
#Discriminator Weights and Biases
W1_d = tf.Variable(tf.random_uniform([1,HLAYER_SIZE],minval=0,maxval=1,dtype = tf.float32))
b1_d = tf.Variable(tf.random_uniform([HLAYER_SIZE],minval=0,maxval=1,dtype = tf.float32))
W2_d = tf.Variable(tf.random_uniform([HLAYER_SIZE,1],minval=0,maxval=1,dtype = tf.float32))
b2_d = tf.Variable(tf.random_uniform([1],minval=0,maxval=1,dtype = tf.float32))
#Generator Weights and Biases
W1_g = tf.Variable(tf.random_uniform([1,HLAYER_SIZE],minval=0,maxval=1,dtype = tf.float32))
b1_g = tf.Variable(tf.random_uniform([HLAYER_SIZE],minval=0,maxval=1,dtype = tf.float32))
W2_g = tf.Variable(tf.random_uniform([HLAYER_SIZE,1],minval=0,maxval=1,dtype = tf.float32))
b2_g = tf.Variable(tf.random_uniform([1],minval=0,maxval=1,dtype = tf.float32))

W_g = [W1_g,W2_g]	#List of Generator Weights
b_g = [b1_g,b2_g]	#List of Generator Biases
W_d = [W1_d,W2_d]	#List of Discriminator Weights
b_d = [b1_d,b2_d]	#List of Discriminator Biases

theta_d = [W1_d,W2_d,b1_d,b2_d]
theta_g = [W1_g,W2_g,b1_g,b2_g]

#Drawing samples
Z = np.random.uniform(0,20,(TOTAL_SAMPLE_SIZE,1))

X = np.random.normal(mu,sigma,(TOTAL_SAMPLE_SIZE,1))
print ('X.size=%d' %X.size) # 1000
print ('Z.size=%d' %Z.size) # 1000
#Define Generator and Discriminator Losses
x = tf.placeholder(tf.float32, shape=(None, 1))
z = tf.placeholder(tf.float32, shape=(None, 1))
D1 = discriminator(x, W_d, b_d) # Tensor("Sigmoid:0", shape=(?, 1), dtype=float32)

D2 = discriminator(generator(z, W_g, b_g), W_d, b_d)
loss_d = tf.reduce_mean(-tf.log(D1) - tf.log(1 - D2))
loss_g = tf.reduce_mean(-tf.log(D2))

#Defining Optimizer
train_step_g = tf.train.AdamOptimizer().minimize(loss_g, var_list = theta_g)
train_step_d = tf.train.AdamOptimizer().minimize(loss_d, var_list = theta_d)

#Training
present_epoch = 0
with tf.Session() as sess:
	tf.global_variables_initializer().run()	
	for step in range(1,TOTAL_STEPS):

		# TOTAL_STEPS = (TOTAL_SAMPLE_SIZE/BATCH_SIZE)*EPOCHS
		# BATCH_SIZE = 10
		# TOTAL_SAMPLE_SIZE = 1000
		present_epoch = int(((step-1)*BATCH_SIZE)/TOTAL_SAMPLE_SIZE)
		
		begin_point = ((step-1)*BATCH_SIZE) % TOTAL_SAMPLE_SIZE
		end_point 	= (step*BATCH_SIZE) % TOTAL_SAMPLE_SIZE
		#print('%d, %d' % (begin_point, end_point))
		x_batch = X[begin_point:end_point,:]
		z_batch = Z[begin_point:end_point,:]
		
		#print x_batch.size # 10
		#Perform a training step
		sess.run(train_step_d, feed_dict={x: x_batch, z: z_batch})
		sess.run(train_step_g, feed_dict={x: x_batch, z: z_batch})
		
		if step % 100 == 0:
			print ('step %d / %d : Epoch:%d' % (step, TOTAL_STEPS, present_epoch))
			
		if step % 1000 == 0:
			data = sess.run(generator(Z,W_g,b_g)) #Saving final distribution
			#print ('step %d / %d : Epoch:%d' % (step, TOTAL_STEPS, present_epoch))
			plot1 = sns.distplot(data, hist=False, rug=True)
			title1 = 'training_epochs = '+str(present_epoch);
			plt.title(title1)
			#plt.ylim(0, 7)
			#plt.xlim(0, 7)
			plt.savefig('out/{}.png'.format(str(present_epoch).zfill(3)), bbox_inches='tight')
			plt.clf()
	data = sess.run(generator(Z,W_g,b_g)) #Saving final distribution

#Plotting
title1 = 'training_epochs = '+str(EPOCHS);
plot1 = sns.distplot(data, hist=False, rug=True)
plt.title(title1)
#plt.savefig('out/{}.png'.format(str(EPOCHS).zfill(3)), bbox_inches='tight')
plt.show()
