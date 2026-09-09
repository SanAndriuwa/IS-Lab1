% Step-by-step perceptron example.
% One object is calculated manually before using a training loop.
clear; clc; close all;

data = dlmread(fullfile(fileparts(mfilename('fullpath')), '..', 'main', 'Data.txt'), ',');
x = data(1,1:2)';       % color and roundness of the first object
T = data(1,3);          % desired class: apple = 1, pear = -1

w = [0.1; -0.2];        % one weight per feature
b = 0;                   % bias
eta = 0.1;               % learning rate

% 1. Weighted sum: combine both features.
v = w' * x + b;

% 2. Step activation: convert the sum to a class.
y = 1;
if v <= 0
    y = -1;
end

% 3. Error: desired class minus current prediction.
e = T - y;

fprintf('Weighted sum v = %.4f\n', v);
fprintf('Prediction y = %d, target T = %d, error e = %d\n', y, T, e);

% 4. Update only if the object was classified incorrectly.
if e ~= 0
    w = w + eta*e*x;
    b = b + eta*e;
end

fprintf('Updated weights: [%.4f %.4f]\n', w(1), w(2));
fprintf('Updated bias: %.4f\n', b);

% The complete Lab1 script repeats exactly these steps for all objects
% and all epochs until there are no classification mistakes.
