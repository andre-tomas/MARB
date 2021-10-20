clear all;
syms chi xi theta phi real
syms g1 g2

c1 = exp(1i*chi)*sin(theta)*cos(phi);
c2 = exp(1i*xi)*sin(theta)*sin(phi);
c3 = cos(theta);

N1kvad = abs(c1)^2/(1-abs(c1)^2);
N2kvad = 1/(1-abs(c1)^2);

A = [abs(c1)^2/2, conj(c1)*c2, conj(c1)*c3;
    0, abs(c2)^2/2, conj(c2)*c3; 
    0, 0, abs(c3)^2/2]; 
A = simplify(A + ctranspose(A)); 

x = (c1 - 1/conj(c1));
B = [abs(x)^2/2, conj(x)*c2, conj(x)*c3; 
    0, abs(c2)^2/2, conj(c2)*c3;
    0, 0, abs(c3)^2/2];
B = simplify(exp(1j*g1)*N1kvad*(B + ctranspose(B)));


C = [0, 0, 0;
     0, abs(c3)^2/2, -conj(c2)*c3
     0, 0, abs(c2)^2/2];
 C = simplify(exp(1j*g2)*N2kvad*(C + ctranspose(C)));
 
 U = simplify(A + B + C)
 
 simplify(det(U))
 simplify(det(U*U))