import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
         (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
         (17, 18), (18, 19), (19, 20), (20, 21),
         (22, 23), (23, 24), (24, 25), (25, 26),
         (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35),
         (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
         (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
         (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
         (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
         (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67,60)]

face_models_num = 30

def plot_face(plt,X,edges,color='b'):
    plt.plot(X[:,0], X[:,1], 'o', color=color)
    for i,j in edges:
        xi,yi = X[i]
        xj,yj = X[j]
        plt.plot((xi,xj), (yi,yj), '-', color=color)
        plt.axis('square')
        plt.xlim(-500,-200)
        plt.ylim(-500,-200)

def make_faces(number_of_faces): # returns arrays of shape (number_of_faces,68,2)
    faces = np.zeros((number_of_faces,68,2))
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:\KNTU\Term7\Linear Algebra\project\shape_predictor_68_face_landmarks.dat')

    i = 0
    index = 0

    while rval:
        rval, img = vc.read()
        i += 1
        if i%3 != 1:
            continue
        else:
            dets = detector(img, 1)
        
        for k, d in enumerate(dets):
            shape = predictor(img, d)
            X = np.array([(shape.part(i).x,shape.part(i).y) for i in range(shape.num_parts)])  
            for x in X:
                cv2.circle(img, (x[0], x[1]), 2, (0,0,255))

        cv2.imshow('', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if index == number_of_faces - 1:
                faces[index,:,:] = -X 
                plot_face(plt, faces[index,:,:], edges, color='b')
                plt.show()
                break
            else:
                faces[index,:,:] = -X 
                plot_face(plt, faces[index,:,:], edges, color='b')
                index += 1
                plt.show()

    return faces

def init_faces():
    faces = make_faces(face_models_num)
    with open('D:/KNTU/Term7/Linear Algebra/project/face_data.txt','w') as f:
        for i in range(0, face_models_num): 
            np.savetxt(f, faces[i,:,:].ravel())

def get_faces():
    with open('D:/KNTU/Term7/Linear Algebra/project/face_data.txt', 'r') as f:
        faces = (np.loadtxt(f)).reshape((face_models_num,68,2))
    return faces

def get_face(filename):
    with open(filename, 'r') as f:
        face = (np.loadtxt(f)).reshape((68,2))
    return face

def draw_faces():
    faces = get_faces()
    for i in range(0, face_models_num):
        plt.cla()
        plot_face(plt, faces[i], edges, color='b')
        plt.draw()
        plt.pause(.5)

# init_faces()

