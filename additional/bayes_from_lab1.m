function y = bayes_from_lab1(P, T, Ptest)
% Reuse features already calculated in the first task.
% P: 2 x N, one training fruit per column (color; roundness).
% T: target classes 1 or -1, N elements.
% Ptest: 2 x M, features of objects to classify.
% Example: y = bayes_from_lab1(P, T, Ptest);

T = T(:);
assert(size(P, 1) == 2 && size(Ptest, 1) == 2, ...
    'P and Ptest must have two rows: color and roundness.');
assert(size(P, 2) == length(T), 'The number of columns in P must equal length(T).');
assert(all(isfinite(P(:))) && all(isfinite(Ptest(:))), 'Features must be finite numbers.');
assert(all(T == 1 | T == -1), 'Classes must be 1 or -1.');
assert(sum(T == 1) >= 2 && sum(T == -1) >= 2, ...
    'Training requires at least two objects from each class.');

classes = [1 -1];
mu = zeros(2, 2);
sigma2 = zeros(2, 2);
prior = zeros(1, 2);
for c = 1:2
    examples = P(:, T == classes(c))';
    mu(c, :) = mean(examples, 1);
    sigma2(c, :) = max(var(examples, 1, 1), 1e-8);
    prior(c) = size(examples, 1) / length(T);
end

y = zeros(size(Ptest, 2), 1);
for i = 1:size(Ptest, 2)
    score = zeros(1, 2);
    for c = 1:2
        score(c) = log(prior(c));
        for j = 1:2
            score(c) = score(c) - 0.5*log(2*pi*sigma2(c, j)) ...
                - (Ptest(j, i)-mu(c, j))^2/(2*sigma2(c, j));
        end
    end
    [~, c] = max(score);
    y(i) = classes(c);
end
end

