import numpy as np

def similarity(neutral_face, other_face): # faces are ndarrays with the shape of (68,2)
    xi = neutral_face[:,0] # shape = (68,1)
    yi = neutral_face[:,1] # shape = (68,1)
    xi_p = other_face[:,0] # shape = (68,1)
    yi_p = other_face[:,1] # shape = (68,1)
    sigma_xi_p_2 = np.dot(xi_p, xi_p) # shape = (1,1)
    sigma_yi_p_2 = np.dot(yi_p, yi_p) # shape = (1,1)
    sigma_xi_xi_p = np.dot(xi, xi_p) # shape = (1,1)
    sigma_yi_yi_p = np.dot(yi, yi_p) # shape = (1,1)
    sigma_xi_p_yi = np.dot(xi_p, yi) # shape = (1,1)
    sigma_xi_yi_p = np.dot(xi, yi_p) # shape = (1,1)
    divisor = sigma_xi_p_2 + sigma_yi_p_2
    a_dividend = sigma_xi_xi_p + sigma_yi_yi_p
    b_dividend = sigma_xi_p_yi - sigma_xi_yi_p 
    a = a_dividend / divisor
    b = b_dividend / divisor
    A = np.array([[a,-b],[b,a]])
    return A

def affine(neutral_face, other_face): # faces are ndarrays with the shape of (68,2)
    xi = neutral_face[:,0] # shape = (68,1)
    yi = neutral_face[:,1] # shape = (68,1)
    xi_p = other_face[:,0] # shape = (68,1)
    yi_p = other_face[:,1] # shape = (68,1)
    sigma_xi_p_2 = np.dot(xi_p, xi_p) # shape = (1,1)
    sigma_yi_p_2 = np.dot(yi_p, yi_p) # shape = (1,1)
    sigma_xi_xi_p = np.dot(xi, xi_p) # shape = (1,1)
    sigma_yi_yi_p = np.dot(yi, yi_p) # shape = (1,1)
    sigma_yi_xi_p = np.dot(xi_p, yi) # shape = (1,1)
    sigma_xi_yi_p = np.dot(xi, yi_p) # shape = (1,1)
    sigma_xi_p_yi_p = np.dot(xi_p,yi_p)
    m1 = sigma_xi_p_yi_p / sigma_yi_p_2
    m2 = sigma_xi_p_yi_p / sigma_xi_p_2
    divisor = 1 - m2 * m1
    b_dividend = (sigma_xi_yi_p / sigma_yi_p_2) - (sigma_xi_xi_p / sigma_xi_p_2) * m1
    b = b_dividend / divisor
    d_dividend = (sigma_yi_yi_p / sigma_yi_p_2) - (sigma_yi_xi_p / sigma_xi_p_2) * m1
    d = d_dividend / divisor
    a = (sigma_xi_xi_p / sigma_xi_p_2) - b * m2
    c = (sigma_yi_xi_p / sigma_xi_p_2) - d * m2
    A = np.array([[a,b],[c,d]])
    return A
