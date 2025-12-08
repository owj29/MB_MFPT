import numpy as np

T = np.loadtxt("/Users/student/Desktop/transition_matrix_0.005_lag.csv", delimiter=",")

def compute_mfpt(transition_matrix, target_state):
    n_states = transition_matrix.shape[0]
    
    # Create indices excluding the target state
    indices = np.arange(n_states)
    mask = indices != target_state
    other_states = indices[mask]
    
    # Extract submatrix excluding target state row and column
    P_sub = transition_matrix[np.ix_(other_states, other_states)]
    
    # Create identity matrix for the subsystem
    I_sub = np.eye(len(other_states))
    
    # Solve the linear system: (I - P_sub) * m = 1
    # This gives mfpt from all states (except target) to target
    A = I_sub - P_sub
    b = np.ones(len(other_states))
    
    # Solve the system
    mfpt_other = np.linalg.solve(A, b)
    
    # Create full MFPT array
    mfpt = np.zeros(n_states)
    mfpt[other_states] = mfpt_other
    mfpt[target_state] = 0.0
    
    return mfpt

mfpt = compute_mfpt(T, 0)
print(mfpt)
print(T.sum(axis=1))