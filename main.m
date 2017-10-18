clear all

input_data = csvread('wine.csv');
classes = unique(input_data(:,1));
k = size(classes,1);
p = size(input_data,1);
n = size(input_data,2) - 1;
is_test = rand(1,p);

learn = input_data(is_test < 0.7,:);
test = input_data(is_test >= 0.7,:);

p = size(learn,1);

x = learn(:,2:n+1);
x(:,n+1) = 1;
x_class = learn(:,1);

eta = 0.01;

w = zeros(k,n+1);
max_iter = 1000000;
i = 0;
dw_max = inf;

while (i < max_iter) && (dw_max > 1e-2)
    t = repmat(x_class',k,1) == repmat(classes,1,p);
    y = sigmoid(w*x');
    nabla_E = (y-t)*x;
    delta_w = -eta*nabla_E;
    dw_max = max(max(abs(delta_w)));
    w = w + delta_w;
    i = i+1;
end

xt = test(:,:);
xt(:,n+2) = 1;
correct = sum(arrayfun(@(i) (classify(xt(i,2:n+2), w, classes) == xt(i,1)),1:size(xt,1)));

fprintf("%d out of %d correct.", correct, size(test,1));
