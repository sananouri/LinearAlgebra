from face_data import edges, plot_face, get_faces, face_models_num
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

(U, S, _) = np.linalg.svd(Z)

s_k = S[0:k]
u_k = U[:,0:k]

for i in range(0, k):
    plt.cla()
    print("sigma", i+1, "=" ,s_k[i])
    for a in np.linspace(-s_k[i], s_k[i], 10):
        X = face_avg.copy().ravel()
        X += a * u_k[:,i]
        plt.cla()
        plot_face(plt, X.reshape((68,2)), edges, color='b')
        plt.draw()
        plt.pause(.1)
    plt.show()

