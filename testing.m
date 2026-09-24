clear;
clc;
close all;

%% 1. Duomenys
folder = fileparts(mfilename('fullpath'));
data = dlmread(fullfile(folder, 'main', 'Data.txt'), ',');

x1 = data(:, 1);       % Spalva
x2 = data(:, 2);       % Apvalumas
target = data(:, 3);   % Obuolys = 1, kriause = -1

epoch = 1000;
eta = 0.1;

%% 2. Pradiniai svoriai ir poslinkis
w1 = 0.1;
w2 = 0.1;
b = 0;

%% 3. Mokymas
for j = 1:epoch
    for i = 1:length(target)

        % Svertine ivesties suma
        v = x1(i)*w1 + x2(i)*w2 + b;

        % Slenkstine aktyvavimo funkcija
        if v > 0
            y = 1;
        else
            y = -1;
        end

        % Klaida
        e = target(i) - y;

        % Perceptrono mokymo taisykle
        w1 = w1 + eta*e*x1(i);
        w2 = w2 + eta*e*x2(i);
        b = b + eta*e;
    end
end

%% 4. Patikrinimas su galutiniais svoriais
% Cia tikrinami mokymo duomenys, ne nauji objektai.
Y = zeros(size(target));

for i = 1:length(target)
    v = x1(i)*w1 + x2(i)*w2 + b;

    if v > 0
        Y(i) = 1;
    else
        Y(i) = -1;
    end
end

disp('Stulpeliai: tikroji klase, prognozuota klase');
disp([target Y]);

%% 5. Rezultatu grafikas
figure;
plot(1:length(target), target, 'o');
hold on;
plot(1:length(target), Y, '*');
xlabel('Objekto numeris');
ylabel('Klase');
legend('Target', 'Perceptronas');
grid on;
