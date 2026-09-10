clear;
clc;
close all;
folder = fileparts(mfilename('fullpath'));
data = dlmread(fullfile(folder, 'main', 'Data.txt'), ',');
x1 = data(:,1); % Color
x2 = data(:,2); % Roundness
target = data(:,3); % Apple = 1, pear = -1

%% Training examples: three apples and two pears
apple = [1 2 3];
pear = [10 11];
test = [4 5 6 7 8 9 12 13];

%% Learn the mean and variance of each feature
m1_apple = mean(x1(apple));
m2_apple = mean(x2(apple));
v1_apple = max(var(x1(apple),1), 1e-8);
v2_apple = max(var(x2(apple),1), 1e-8);
m1_pear = mean(x1(pear));
m2_pear = mean(x2(pear));
v1_pear = max(var(x1(pear),1), 1e-8);
v2_pear = max(var(x2(pear),1), 1e-8);
% A small variance floor prevents division by zero.

%% Prior probability of each class
p_apple = length(apple)/(length(apple)+length(pear));
p_pear = length(pear)/(length(apple)+length(pear));

%% Testing: Gaussian density for each feature
Y = zeros(size(test));
for i = 1:length(test)
    k = test(i);
    f1_apple = exp(-(x1(k)-m1_apple)^2/(2*v1_apple))/sqrt(2*pi*v1_apple);
    f2_apple = exp(-(x2(k)-m2_apple)^2/(2*v2_apple))/sqrt(2*pi*v2_apple);
    f1_pear = exp(-(x1(k)-m1_pear)^2/(2*v1_pear))/sqrt(2*pi*v1_pear);
    f2_pear = exp(-(x2(k)-m2_pear)^2/(2*v2_pear))/sqrt(2*pi*v2_pear);

    % Naive Bayes treats the features as independent within each class.
    % These are comparison scores, not normalized probabilities.
    score_apple = p_apple*f1_apple*f2_apple;
    score_pear = p_pear*f1_pear*f2_pear;
    if score_apple >= score_pear
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end
target_test = target(test);
fprintf('Test accuracy: %.1f%%\n', 100*mean(Y(:) == target_test(:)));
disp('Columns: object number, target, prediction');
disp([test(:) target_test(:) Y(:)]);
figure;
plot(test, target_test, 'o');
hold on;
plot(test, Y, '*');
legend('Target', 'Bayes');
xlabel('Object number');
ylabel('Class: apple = 1, pear = -1');
