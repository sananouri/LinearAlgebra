from face_data import edges, plot_face, get_faces, face_models_num, get_face
import matplotlib.pyplot as plt
import register
import numpy as np

k = 16
faces = get_faces()

registered_faces = np.zeros((face_models_num,68,2))
registered_faces[0] = faces[0]
face_avg = registered_faces[0].copy()

for i in range(1, face_models_num):
    A = register.similarity(faces[0], faces[i]) 
    registered_faces[i] = (A @ faces[i].T).T
    face_avg += registered_faces[i]

face_avg = face_avg / face_models_num # shape = (68,2)

Z = np.zeros((136, face_models_num))
for i in range(0, face_models_num):
    Z[:,i] = (registered_faces[i] - face_avg).ravel()

(U, _, __) = np.linalg.svd(Z)

u_k = U[:,0:k]

other = get_face('D:/KNTU/Term7/Linear Algebra/project/other_laugh.txt')
A = register.affine(face_avg, other)
other_registered = (A @ other.T).T
(a_optimal,_,__,___) = np.linalg.lstsq(u_k, (other_registered - face_avg).ravel(), rcond = None)
transferred_face = face_avg + (u_k @ a_optimal).reshape((68,2))

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('avg face(g) vs other face(r) vs other face registered (b)')
plot_face(plt, face_avg, edges, color='g')
plot_face(plt, other, edges, color='r')
plot_face(plt, other_registered, edges, color='b')

ax1 = fig.add_subplot(1,2,2)
ax1.set_title('transfered face (r) vs avg face (g)')
plot_face(plt, face_avg, edges, color='g')
plot_face(plt, transferred_face, edges, color='r')

plt.show()
