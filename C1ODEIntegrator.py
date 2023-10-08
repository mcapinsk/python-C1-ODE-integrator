import numpy as np
from scipy.integrate import odeint

# In below function V is a vector which stores both the position of a d-dimensional vector
# and a d x d monodromy matrix. The dimension of V is k=d+d*d. Below function
# computes d from V.
def getDimension(V):
    k=V.size
    return int((-1+np.sqrt(1+4*k))/2)

# V=[x,A] consists of d-dimensional vector x, which is the position in the state space
# and of A, which is a d x d monodromy matrix. Below function returns x from V.
def VectorPart(V):
    d=getDimension(V)
    x=np.array([0.0 for k in range(d)])
    for k in range(d):
        x[k]=V[k]
    return x

# For V=[x,A] below function returns A from V.
def MatrixPart(V):
    d=getDimension(V)
    l=d
    A=np.array([[0.0 for k in range(d)] for j in range(d)])
    for k in range(d):
        for j in range(d):
            A[k][j]=V[l]
            l=l+1
    return A

###################################
# Below function defines a variational ODE
#   x' = f(x,t)
#   A' = Df(x,t)*A
def VariationalVectorField(V,t,f,Df):
    x=VectorPart(V)
    A=MatrixPart(V)
    return np.append(f(x,t),Df(x,t) @ A)

# Below function integrates an ODE with given:
# - vector field f
# - derivative of the vector field Df
# - initial condition x, which is a vector
# - set of times T, which is an array T=[t0,t1,...,tn] of times at which the
#   solution of the ode is to be computed.
def SolveVariationalODE(f,Df,x,T):
    V=np.append(x,np.eye(x.size))
    return odeint(VariationalVectorField,V,T,args=(f, Df))

###########################################
# Example of application:
###########################################

# Below function computes the derivative of the flow.
def DPhi(f,Df,x,t):
    return MatrixPart(SolveVariationalODE(f,Df,x,[0.0,t])[1])

# Below function computes the flow:
def Phi(f,x,t):
    return odeint(f,x,[0.0,t])[1]

# Here is how to use this for the Leorenz system.
# The Lorenz system is autonomous, but we add the time variable since our code
# was developed to allow for time dependent ODEs.
def Lorenz(x,t):
    sigma=10.0
    beta=8.0/3.0
    rho=28.0
    v=np.array([0.0 for i in range(3)])
    v[0]=sigma*(x[1]-x[0])
    v[1]=x[0]*(rho-x[2])-x[1]
    v[2]=x[0]*x[1]-beta*x[2]
    return v

def DLorenz(x,t):
    sigma=10.0
    beta=8.0/3.0
    rho=28.0
    A=np.array([[0.0 for i in range(3)] for j in range(3)])

    A[0][0]=-sigma
    A[0][1]=sigma
    A[0][2]=0.0

    A[1][0]=rho-x[2]
    A[1][1]=-1.0
    A[1][2]=-x[0]

    A[2][0]=x[1]
    A[2][1]=x[0]
    A[2][2]=-beta
    return A

def Example1():
    print("\nExample 1: ")
    x=np.array([1.0,2.0,3.0])
    t=1.234
    print("The initial condition is x = ", x)
    print("Phi(x,t)  = ", Phi(Lorenz,x,t))
    print("DPhi(x,t) = \n", DPhi(Lorenz,DLorenz,x,t))

###########################################
# Above approach requires us to write the formula for the derivative of the vector field,
# namely we need to have the function Df(x,t).
# If you are too lazy to type this in, below is an alternative, which uses only the
# vector field, namely only the function f(t,x).
# (Below approach is based on a crude numerical approximation of the derivative
# of the flow, so the earlier approach, where we need provides the explicit formula for
# the derivative Df(x,t) of the vector field is better.)

def Derivative(f,x,t,h):
    d=x.size
    A=np.array([[0.0 for k in range(d)] for j in range(d)])
    for k in range(d):
        y=np.array(x)
        y[k]=y[k]+h
        v=(1.0/h)*(f(y,t)-f(x,t))
        for j in range(d):
            A[j][k]=v[j]
    return A

def VariationalVectorField2(V,t,f,h):
    x=VectorPart(V)
    A=MatrixPart(V)
    return np.append(f(x,t),Derivative(f,x,t,h) @ A)

def SolveVariationalODE2(f,x,T,h):
    V=np.append(x,np.eye(x.size))
    return odeint(VariationalVectorField2,V,T,args=(f, h))

############################################
# here is an example of application
def Dphi(f,x,t):
    # Below parameter h is used for the numerical approximation of the derivative
    # of the flow.
    h=0.00001
    return MatrixPart(SolveVariationalODE2(f,x,[0,t],h)[1])

############################################
def Example2():
    print("\nExample 2: ")
    x=np.array([1.0,2.0,3.0])
    t=1.234
    print("The initial condition is x = ", x)
    print("Phi(x,t)  = ", Phi(Lorenz,x,t))
    print("Dphi(x,t) = \n", Dphi(Lorenz,x,t))

##############################################
# Below is a simple class that can be used to integrate an ODE

class C1Flow:
    def __init__(self,f,Df):
        self.f=f
        self.Df=Df
        self.Derivative=0
        self.Position=0
    def __call__(self,x,t):
        V=SolveVariationalODE(self.f,self.Df,x,[0.0,t])[1]
        self.Derivative=MatrixPart(V)
        self.Position=VectorPart(V)
        return self.Position

def Example3():
    print("\nExample 3: ")
    x=np.array([1.0,2.0,3.0])
    t=1.234
    Phi=C1Flow(Lorenz,DLorenz)
    print("The initial condition is x = ", x)
    print("Phi(x,t)  = ", Phi(x,t))
    print("DPhi(x,t) = \n", Phi.Derivative)

##############################################
# Here is another class that can be used for the approach where we
# do not provide the formula for the derivative of the vector field.
class C1Flow2:
    def __init__(self,f,h):
        self.f=f
        self.h=h
        self.Derivative=0
        self.Position=0
    def __call__(self,x,t):
        V=SolveVariationalODE2(self.f,x,[0.0,t],self.h)[1]
        self.Derivative=MatrixPart(V)
        self.Position=VectorPart(V)
        return self.Position

def Example4():
    print("\nExample 4: ")
    x=np.array([1.0,2.0,3.0])
    t=1.234
    h=0.0001
    Phi=C1Flow2(Lorenz,h)
    print("The initial condition is x = ", x)
    print("Phi(x,t)  = ", Phi(x,t))
    print("DPhi(x,t) = \n", Phi.Derivative)
