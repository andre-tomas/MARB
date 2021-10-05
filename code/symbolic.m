clear all;
syms chi xi theta phi real

c1 = exp(1i*chi)*sin(theta)*cos(phi);
c2 = exp(1i*xi)*sin(theta)*sin(phi);
c3 = cos(theta);

A = [abs(c1)^2/2, conj(c1)*c2, conj(c1)*c3;
    0, abs(c2)^2/2, conj(c2)*c3; 
    0, 0, abs(c3)^2/2]; 
A = simplify(A + ctranspose(A));

x = (c1 - 1/conj(c1));
B = [abs(x)^2/2, conj(x)*c2, conj(x)*c3; 
    0, abs(c2)^2/2, conj(c2)*c3;
    0, 0, abs(c3)^2/2];
B = simplify((abs(c1)/(1 - abs(c1)))*(B + ctranspose(B)));


C = [0, 0, 0;
     0, abs(c3)^2/2, -conj(c2)*c3
     0, 0, abs(c2)^2/2];
 C = simplify((1/(1 - abs(c1)))*(C + ctranspose(C)));
 
 U = simplify(A + B + C)