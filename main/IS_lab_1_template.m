% Perceptron classification using two image features.
% The features in Data.txt were extracted from the apple and pear images
% with spalva_color.m and apvalumas_roundness.m.
clear; clc; close all;

folder = fileparts(mfilename('fullpath'));
data = dlmread(fullfile(folder, 'Data.txt'), ',');
X = data(:, 1:2);        % rows: objects, columns: color and roundness
T = data(:, 3);          % apple = 1, pear = -1

rng(1);                  % repeatable random initialization
eta = 0.1;               % learning rate
maxEpochs = 1000;

w = randn(2, 1);         % two weights, one for each feature
b = randn;               % bias (threshold shift)
errorsPerEpoch = zeros(maxEpochs, 1);

for epoch = 1:maxEpochs
    errors = 0;

    for i = 1:size(X, 1)
        % Weighted sum: v = w' * x + b.
        v = w' * X(i, :)' + b;

        % Step activation function.
        y = 1;
        if v <= 0
            y = -1;
        end

        % Perceptron error and update.
        e = T(i) - y;
        if e ~= 0
            w = w + eta * e * X(i, :)';
            b = b + eta * e;
            errors = errors + 1;
        end
    end

    errorsPerEpoch(epoch) = errors;
    if errors == 0
        break;             % all training objects are classified correctly
    end
end

errorsPerEpoch = errorsPerEpoch(1:epoch);

% Final predictions.
v = X * w + b;
Y = ones(size(T));
Y(v <= 0) = -1;
accuracy = 100 * mean(Y == T);

fprintf('Epochs: %d\n', epoch);
fprintf('Training accuracy: %.1f%%\n', accuracy);
fprintf('Final weights: w1 = %.4f, w2 = %.4f, b = %.4f\n', ...
    w(1), w(2), b);

% Plot feature points and the learned decision boundary.
figure;
hold on;
scatter(X(T == 1, 1), X(T == 1, 2), 60, 'r', 'filled');
scatter(X(T == -1, 1), X(T == -1, 2), 60, 'b', 'filled');

xLine = linspace(min(X(:, 1)), max(X(:, 1)), 100);
if abs(w(2)) > eps
    yLine = -(w(1) * xLine + b) / w(2);
    plot(xLine, yLine, 'k-', 'LineWidth', 1.5);
else
    xline(-b / w(1), 'k-', 'LineWidth', 1.5);
end

grid on;
xlabel('Color');
ylabel('Roundness');
legend('Apple (1)', 'Pear (-1)', 'Decision boundary', 'Location', 'best');
title('Perceptron: apples and pears');

figure;
stairs(1:epoch, errorsPerEpoch, 'LineWidth', 1.5);
grid on;
xlabel('Epoch');
ylabel('Number of mistakes');
title('Perceptron learning progress');
