import math
class  Value:
    
    def __init__(self,data,_children=(),_op='') :
        self.data=data 
        self.grad=0.0 # NOTE : We chose 0 because we suppose that this variable has no effect on the output.
        
        # internal variables used for autograd graph construction
        self._backward=lambda:None # NOTE : this is function that make the calcul of the gradient
        self._prev=set(_children) # children
        self._op=_op # NOTE : operation used for graphviz / debugging / etc

    ### operations -----------------------------------------------------------------------------------------: 
    def __add__(self, other): # self + other
        other=other if isinstance(other,Value) else Value(other) 
        out=Value(self.data+other.data,(self,other),'+') 
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self, other): # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other): # self * other
        other=other if isinstance(other,Value) else Value(other) 
        
        out = Value(self.data*other.data,(self,other),'*') 
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    
    ### functions -----------------------------------------------------------------------------------------:
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out
    
    def tanh(self):
        out =Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,),'tanh')
        
        def _backward():
            self.grad += (1-out.data**2)*out.grad
        
        out._backward=_backward
        
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    ### backward -----------------------------------------------------------------------------------------:
    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    
    ###-----------------------------------------------------------------------------------------:
    def __neg__(self): # -self => self*(-1)
        return self * -1
    
    def __sub__(self, other): # self - other => self + (-other)
        return self + (-other)
    
    def __truediv__(self, other): # self / other => self * (other**-1)
        return self * other**-1
    
    def __radd__(self,other): # other + self 
        return self+other
    
    def __rsub__(self, other): # other - self
        return other + (-self)
    
    def __rmul__(self,other): # other * self 
        return self*other
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    
    
    