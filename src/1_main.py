from face_data import edges, plot_face, get_faces, face_models_num
import matplotlib.pyplot as plt
import register
import numpy as np

faces = get_faces() # shape = (n, 68, 2)

registered_faces = np.zeros((face_models_num, 68, 2))
registered_faces[0] = faces[0].copy()
face_avg = registered_faces[0].copy()

for i in range(1, face_models_num):
    A = register.similarity(faces[0], faces[i]) 
    registered_faces[i] = (A @ faces[i].T).T
    face_avg += registered_faces[i]

face_avg = face_avg / face_models_num

# for i in range(0,face_models_num):
i = 4
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.set_title('avg face(g) vs new face(r) vs neutral (b)')
plot_face(plt, registered_faces[0,:,:], edges, color='b')
plot_face(plt, registered_faces[i,:,:], edges, color='r')
plot_face(plt, face_avg, edges, color='g')

ax1 = fig.add_subplot(2,3,2)
ax1.set_title('old(g) vs new face(r) vs neutral (b)')
plot_face(plt, registered_faces[0,:,:], edges, color='b')
plot_face(plt, registered_faces[i,:,:], edges, color='r')
plot_face(plt, faces[i,:,:], edges, color='g')

ax1 = fig.add_subplot(2,3,4)
ax1.set_title('neutral (b)')
plot_face(plt, registered_faces[0,:,:], edges, color='b')

ax1 = fig.add_subplot(2,3,5)
ax1.set_title('new face (r)')
plot_face(plt, registered_faces[i,:,:], edges, color='r')

ax1 = fig.add_subplot(2,3,6)
ax1.set_title('avg face (g)')
plot_face(plt, face_avg, edges, color='g')

plt.show()

