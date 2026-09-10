clear;
clc;
close all;

%% Training data: color, roundness and class
folder = fileparts(mfilename('fullpath'));
data = dlmread(fullfile(folder, 'main', 'Data.txt'), ',');
x1 = data(:,1);
x2 = data(:,2);
target = data(:,3); % Apple = 1, pear = -1
epoch = 1000;
eta = 0.1;

%% Weights and bias
w1 = randn(1);
w2 = randn(1);
b = randn(1);

%% Training
for j = 1:epoch
    errors = 0;
    for i = 1:length(target)
        % Forward pass
        v = x1(i)*w1 + x2(i)*w2 + b;
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % Error
        e = target(i) - y;
        if e ~= 0
            errors = errors + 1;
        end

        % Update weights and bias
        w1 = w1 + eta*e*x1(i);
        w2 = w2 + eta*e*x2(i);
        b = b + eta*e;
    end
    if errors == 0
        break;
    end
end

%% Check all training objects with the final weights
Y = zeros(size(target));
for i = 1:length(target)
    v = x1(i)*w1 + x2(i)*w2 + b;
    if v > 0
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end
disp('Columns: target, prediction');
disp([target Y]);
fprintf('Training accuracy: %.1f%%\n', 100*mean(Y == target));

%% Feature plot and decision boundary
figure;
plot(x1(target == 1), x2(target == 1), 'ro');
hold on;
plot(x1(target == -1), x2(target == -1), 'b*');
x_line = linspace(min(x1), max(x1), 100);
if abs(w2) > eps
    y_line = -(w1*x_line + b)/w2;
    plot(x_line, y_line, 'k-');
elseif abs(w1) > eps
    xline(-b/w1);
end
xlabel('Color');
ylabel('Roundness');
legend('Apple', 'Pear', 'Decision boundary');
title('Training objects');
grid on;
