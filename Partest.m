%Create a parallel cluster object
c = parcluster;
%Specify the number of cores for the cluster object
poolObject = parpool(c,6);
%Start timing
tic
n = 30;
A = 60;
a = zeros(n);
parfor i = 1:n*n
 a(i) = max(abs(eig(rand(A))));
end
disp(a)
%End timing
time = toc;
fprintf('The parallel method is executed in %5.2f seconds. \n',time);
%Delete the parallel object
delete(poolObject);
