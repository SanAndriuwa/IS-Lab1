% Additional task: Naive Bayes classifier.
% Two features: color and roundness. Apple = 1, pear = -1.
clear; clc; close all;

folder = fileparts(mfilename('fullpath'));
data = load(fullfile(folder, 'Data.txt'));
X = data(:, 1:2);
d = data(:, 3);

% As in the original template: train on three apples and two pears.
train = [1 2 3 10 11];
test = [4 5 6 7 8 9 12 13];
classes = [1 -1];
mu = zeros(2, 2);
sigma2 = zeros(2, 2);
prior = zeros(1, 2);

% Find the feature mean and variance for each class.
for c = 1:2
    rows = train(d(train) == classes(c));
    mu(c, :) = mean(X(rows, :), 1);
    sigma2(c, :) = var(X(rows, :), 1, 1);
    sigma2(c, :) = max(sigma2(c, :), 1e-8); % avoid division by zero
    prior(c) = length(rows) / length(train);
end

% Calculate a score for each class and each object.
% Logarithms let us add probabilities instead of multiplying them.
score = zeros(size(X, 1), 2);
for i = 1:size(X, 1)
    for c = 1:2
        score(i, c) = log(prior(c));
        for j = 1:2
            score(i, c) = score(i, c) ...
                - 0.5 * log(2*pi*sigma2(c, j)) ...
                - (X(i, j) - mu(c, j))^2 / (2*sigma2(c, j));
        end
    end
end

y = zeros(size(d));
for i = 1:length(d)
    [~, c] = max(score(i, :));
    y(i) = classes(c);
end

disp('Test objects (1 = apple, -1 = pear):');
disp(table(test', d(test), y(test), ...
    'VariableNames', {'Object', 'Expected', 'Predicted'}));
fprintf('Training accuracy: %.1f %%\n', 100*mean(y(train) == d(train)));
fprintf('Test accuracy: %.1f %%\n', 100*mean(y(test) == d(test)));

figure;
plot(X(d == 1, 1), X(d == 1, 2), 'ro', 'MarkerSize', 8); hold on;
plot(X(d == -1, 1), X(d == -1, 2), 'bs', 'MarkerSize', 8);
plot(X(test, 1), X(test, 2), 'k+', 'MarkerSize', 12);
xlabel('Color'); ylabel('Roundness'); grid on;
legend('Apples', 'Pears', 'Test objects', 'Location', 'best');
title('Naive Bayes: input data');

