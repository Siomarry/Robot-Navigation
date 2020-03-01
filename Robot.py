import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
 
 
np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2
 
 
def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))
 
def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA #if cv2.__version__[0] == '3' else cv2.CV_AA
    color = (r, g, b)
    ctrx = center[0,0]
    ctry = center[0,1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)
    
 
def mouseCallback(event, x, y, flags,null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs
    print("鼠标横坐标位置"+str(x))
    print("鼠标的纵坐标位置"+str(y))
    sensorSigma = 0
    sensorMu = 1
    center=np.array([[x,y]])
    trajectory=np.vstack((trajectory,np.array([x,y])))
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu#Sigma是期望，sensorMu是标准差
    
    if previous_x >0:
        heading=np.arctan2(np.array([y-previous_y]), np.array([previous_x-x ]))
 
        if heading>0:
            heading=-(heading-np.pi)
        else:
            heading=-(np.pi+heading)
            
        distance=np.linalg.norm(np.array([[previous_x,previous_y]])-np.array([[x,y]]) ,axis=1)#向量模长，用上一步的状态和这一步的位置状态求距离,axis按行向量表示
        #计算前进的距离距离
        std=np.array([2,4])
        u=np.array([heading,distance]) #u为输入,前进方向以及距离
        predict(particles, u, std, dt=1.)#预测值
        #各个地标减去中心位置的距离加上一些每一个地标测量时可能会产生的随机误差。
        zs = (np.linalg.norm(landmarks - center, axis=1))#+ (np.random.randn(NL) * sensor_std_err))#真实测量值
        update(particles, weights, z=zs, R=50, landmarks=landmarks)#用真实值更新测量值
        
        indexes = systematic_resample(weights)#通过weights设置重采样各个点的坐标
        resample_from_index(particles, weights, indexes)#重采样,给与新的粒子群其位置，以及对应坐标上的权重
        estimate(particles,weights)
    #状态转移，将鼠标的位置作为上一个位置。
    previous_x=x
    previous_y=y
    
 
 
WIDTH=800
HEIGHT=600
WINDOW_NAME="Particle Filter"
 
#sensorMu=0
#sensorSigma=3
 
sensor_std_err=5
 
 
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles
def predict(particles, u, std, dt=1.):
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])#前进时产生的噪声。
    particles[:, 0] += np.cos(u[0]) * dist#particles的相对横坐标距离
    particles[:, 1] += np.sin(u[0]) * dist#particles的相对纵坐标距离
   
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)#初始化权重
    for i, landmark in enumerate(landmarks):
        distance=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)#计算地标点和粒子之间的距离#1*400 tuple
        #weights *= scipy.stats.pareto(min(distance),distance).pdf(z[i])#对应z[i]的概率值,#weights代表各个位置的权重，各个位置的权重由真实值distance形成的正态概率分布确定.#所有地标的权重乘积
        weights *= scipy.stats.norm(distance,R).pdf(z[i])
    weights += 1.e-300 # avoid round-off to zero#避免权重过小等于0
    weights /= sum(weights)#权重归一化
    
def neff(weights):
    return 1. / np.sum(np.square(weights))
 
def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N#增加一个噪声，粒子位置分布
 
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    #重新赋予新的位置，将粒子重分布在权重大的区域。
    while i < N and j<N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    print("预测的横坐标为"+(str)(mean))
    posy = particles[:,1]
    meany = np.average(posy,weights=weights,axis=0)
    #partical = np.arange(2)
    #for i in particles:
    #    curdistance = abs(i[0] - mean[0])
    #    partical = particles[0]
    #    if abs(i[0] - mean[0]) <abs(curdistance):
    #        partical = i
    #print("预测的纵坐标为"+str(partical[1]))
    print("预测的纵坐标为" + str(meany))


    return mean, var
 
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)
 
    
x_range=np.array([0,800])#宽
y_range=np.array([0,600])#高
 
#Number of partciles
N=400#粒子数目
 
landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])#地标位置
NL = len(landmarks)#地标个数
particles=create_uniform_particles(x_range, y_range, N)#给出初始的平均粒子粉不

weights = np.array([1.0]*N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,mouseCallback)
 
center=np.array([[-10,-10]])
 
trajectory=np.zeros(shape=(0,2))
robot_pos=np.zeros(shape=(0,2))
previous_x=-1
previous_y=-1
DELAY_MSEC=50


while(1):
    cv2.imshow(WINDOW_NAME,img)
    img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
    drawLines(img, trajectory,   0,   255, 0)
    drawCross(img, center, r=255, g=0, b=0)
    
    #landmarks
    for landmark in landmarks:
        cv2.circle(img,tuple(landmark),10,(255,0,0),-1)
    
    #draw_particles:
    for particle in particles:
        cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,255),-1)
 
    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break
    
    cv2.circle(img,(10,10),10,(255,0,0),-1)
    cv2.circle(img,(10,30),3,(255,255,255),-1)
    cv2.putText(img,"Landmarks",(30,20),1,1.0,(255,0,0))
    cv2.putText(img,"Particles",(30,40),1,1.0,(255,255,255))
    cv2.putText(img,"Robot Trajectory(Ground truth)",(30,60),1,1.0,(0,255,0))
 
    drawLines(img, np.array([[10,55],[25,55]]), 0, 255, 0)
    
 
 
cv2.destroyAllWindows()