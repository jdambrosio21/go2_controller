import casadi as ca
import numpy as np

def get_pade_coefficients(m):
    """
    Compute coefficients for degree m diagonal Padé approximation to exp(x).
    Returns coefficients for numerator and denominator polynomials.
    """
    c = 1
    p = [c]  # Numerator coeffs
    q = [c]  # Denominator coeffs
    
    for j in range(1, m+1):
        c = c * (m - j + 1) / (j * (2*m - j + 1))
        p.append(c)
        q.append(c if j % 2 == 0 else -c)
        
    return p, q

def matrix_powers(A, max_power):
    """
    Efficiently compute powers of matrix A up to max_power.
    """
    powers = {1: A}
    if max_power >= 2:
        A2 = ca.mtimes(A, A)
        powers[2] = A2
        
    for k in range(3, max_power + 1):
        if k % 2 == 0:
            powers[k] = ca.mtimes(powers[k//2], powers[k//2])
        else:
            powers[k] = ca.mtimes(powers[k-1], A)
            
    return powers

def evaluate_matrix_polynomial(coeffs, A_powers):
    """
    Evaluate matrix polynomial given coefficients and precomputed powers.
    """
    n = A_powers[1].shape[0]
    result = coeffs[0] * ca.DM.eye(n)
    
    for k in range(1, len(coeffs)):
        if coeffs[k] != 0:
            result = result + coeffs[k] * A_powers[k]
            
    return result

def matrix_exp_pade(A, m=13):
    """
    Compute matrix exponential using improved scaling and squaring method.
    
    Args:
        A: CasADi matrix
        m: Degree of Padé approximant (default 13)
    Returns:
        Matrix exponential exp(A)
    """
    # Set scaling parameter (s) based on Higham's analysis
    theta_13 = 5.371920351148152
    s = np.maximum(0, int(np.ceil(np.log2(theta_13))))
    
    # Scale matrix
    A_scaled = A/2**s if s > 0 else A
    
    # Get Padé coefficients
    p_coeffs, q_coeffs = get_pade_coefficients(m)
    
    # Compute required matrix powers
    A_powers = matrix_powers(A_scaled, m)
    
    # Evaluate numerator and denominator
    P = evaluate_matrix_polynomial(p_coeffs, A_powers)
    Q = evaluate_matrix_polynomial(q_coeffs, A_powers)
    
    # Solve matrix equation QR = P
    R = ca.solve(Q, P, 'symbolicqr')
    
    # Squaring phase
    for _ in range(s):
        R = ca.mtimes(R, R)
        
    return R

def test_matrix_exp():
    """
    Test function demonstrating usage.
    """
    # Create test matrix
    n = 2
    A = ca.MX.sym('A', n, n)
    
    # Create function
    exp_func = ca.Function('matrix_exp', [A], [matrix_exp_pade(A)])
    
    # Test with numerical values
    A_test = np.array([[1.0, 2.0], 
                       [-1.0, -2.0]])
    
    # Convert to CasADi DM
    A_test_casadi = ca.DM(A_test)
    
    result = exp_func(A_test_casadi)
    print("Test matrix:")
    print(A_test)
    print("\nMatrix exponential:")
    print(result)
    
    return exp_func

if __name__ == "__main__":
    test_matrix_exp()