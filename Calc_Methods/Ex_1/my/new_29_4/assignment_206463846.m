%---------------------------- General-------------------------------


%---------------------------- Qestion 1-------------------------------
% Data General Defenitions
M = 18;
rho = 1;
q_exact = [2;0;8;8;3;9;8;4;5;2;0;8;8;3;9;8;4;5];
tol = 10^-3;
StrCon = zeros(length(q_exact), 1);

% Plotting General Defenitions
Q1 = figure('Visible', 'on');
movegui(Q1, 'west');

% Gauss-Seidel

%---------------------------- Qestion 1a-------------------------------
% 1a Defenitions

h = (rho * pi) / (M * 5);
A = build_A(h, rho, M ,'sqrt');
v = A * q_exact;

% Gauss-Seidel
[q_gauss_seidel, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon);

% Plot Results
subplot(5,1,1); % 5 rows, 1 column, first plot
semilogy(real_err, '-o');  % First line
hold on;
semilogy(rel_dis, '--*');  % Second line
hold off;

xlabel('Iterations');
ylabel('Error (log)');
title('(Run 1a)');
grid on;



%---------------------------- Qestion 1b-------------------------------

%-------h = (rho * pi) / (M )------------
h = (rho * pi) / (M);
A = build_A(h, rho, M ,'sqrt');
v = A * q_exact;

% Gauss-Seidel
[q_gauss_seidel, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon);

% Plot Results
subplot(5,1,2); % 5 rows, 1 column, first plot
semilogy(real_err, '-o');  % First line
hold on;
semilogy(rel_dis, '--*');  % Second line
hold off;

xlabel('Iterations');
ylabel('Error (log)');
title('(Run 1b - h = (rho * pi) / (M ))');
grid on;

%-------h = (rho * pi) / (M * 2)--------
h = (rho * pi) / (M * 2);
A = build_A(h, rho, M ,'sqrt');
v = A * q_exact;

% Gauss-Seidel
[q_gauss_seidel, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon);

% Plot Results
subplot(5,1,3); % 5 rows, 1 column, first plot
semilogy(real_err, '-o');  % First line
hold on;
semilogy(rel_dis, '--*');  % Second line
hold off;

xlabel('Iterations');
ylabel('Error (log)');
title('(Run 1b - h = (rho * pi) / (M * 2 )');
grid on;

%---------------------------- Qestion 1c-------------------------------

h = (rho * pi) / (M * 5);
A = build_A(h, rho, M ,'sqrt');
v = A * q_exact;

% Jacobi
[q_gauss_seidel, real_err, rel_dis] = jacobi(A, v, q_exact, tol, StrCon);

% Plot Results
subplot(5,1,4); % 5 rows, 1 column, first plot
semilogy(real_err, '-o');  % First line
hold on;
semilogy(rel_dis, '--*');  % Second line
hold off;

xlabel('Iterations');
ylabel('Error (log)');
title('(Run 1c)');
grid on;

%---------------------------- Qestion 1d-------------------------------


h = (rho * pi) / (M * 5);
A = build_A(h, rho, M ,'no_sqrt');
v = A * q_exact;

% Jacobi
[q_gauss_seidel, real_err, rel_dis] = jacobi(A, v, q_exact, tol, StrCon);

% Plot Results
subplot(5,1,5); % 5 rows, 1 column, first plot
semilogy(real_err, '-o');  % First line
hold on;
semilogy(rel_dis, '--*');  % Second line
hold off;

xlabel('Iterations');
ylabel('Error (log)');
title('(Run 1d)');
grid on;


%---------------------------- Qestion 2-------------------------------


%---------------------------- Qestion 2a-------------------------------


%---------------------------- Qestion 2b-------------------------------



%% --- Functions ---


%% ------------------------------Gauss-Seidel------------------------------------------------------------------------------
function [q_current, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon)
    D = diag(diag(A));
    L = tril(A, -1);
    Q = L + D;
    U =  Q - A;
    C = Q \ v;
    G = Q \ (U);
    q_previous = C;
    i = 1;
    err = norm(q_exact-q_previous,'inf');
    rel_dis = [];
    while err > tol && i <1001
        
        q_current = G * q_previous + C;
        real_err(i) = norm(q_current - q_exact,'inf') / norm(q_exact,'inf');

        rel_dis(i) = norm(q_current - q_previous, 'inf') / norm(q_previous, 'inf');
        err = norm(q_exact-q_previous,'inf');
        q_previous = q_current;
        i = i + 1;
    
    end
end
%% ------------------------------Jacobi------------------------------------------------------------------------------

function [q_current, real_err, rel_dis] = jacobi(A, v, q_exact, tol, StrCon)
    D = diag(diag(A));
    L = tril(A, -1);
    Q = L + D;
    U =  A - Q;
    C = D \ v;
    G = -D \ (L+U);
    q_previous = C;
    i = 1;
    err = norm(q_exact-q_previous,'inf');
    rel_dis = [];
    while err > tol && i <1000
        
        q_current = G * q_previous + C;
        real_err(i) = norm(q_current - q_exact,'inf') / norm(q_exact,'inf');

        rel_dis(i) = norm(q_current - q_previous, 'inf') / norm(q_previous, 'inf');
        err = norm(q_exact-q_previous,'inf');
        q_previous = q_current;
        i = i + 1;
    
    end
end
%% ------------------------------Plot Figure------------------------------------------------------------------------------


%% ------------------------------Build A------------------------------------------------------------------------------
function A = build_A(h, rho, M ,mat_calc)
    % Build the matrix A according to the given 3D formula
    A = zeros(M, M);

    for m = 1:M
        for n = 1:M
            if strcmp(mat_calc, 'no_sqrt')
                Rmn = ((h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2);
            else
                Rmn = sqrt((h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2);
            end
            A(m, n) = 1 / (4 * pi * Rmn);
        end
    end
   
end

%% ------------------------------Extras--------------------------------------------------------------


