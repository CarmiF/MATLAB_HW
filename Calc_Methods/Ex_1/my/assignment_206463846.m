% Parameters
M = 18;
rho = 1;
h = (rho * pi) / (5 * M); % As specified in the question

% Build the matrix A
A = build_A(h, rho, M);

% Define the vector q
q = [2;0;6;4;6;3;8;4;6;2;0;6;4;6;3;8;4;6];

% Calculate v = A * q
v = A * q;

% Initialize the starting vector for Gauss-Seidel
StrCon = zeros(length(v), 1);

% Solve using Gauss-Seidel method
[q_new, iterations] = jacobi(A, v, q, 10^(-3), StrCon);

%% --- Functions ---

function [ q_new, iterations] = gauss_seidel(A, v, q_exact, tol, initial_guess)
    % Decompose matrix A into D (diagonal), L (lower), and U (upper) parts
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);

    Q = L + D;
    C = Q \ v;
    G = Q \ (-U);

    q_previous = initial_guess;
    iterations = 1;

    % Initialize error vectors
    relative_error = [];
    relative_error(iterations) = norm(q_exact - q_previous, 'inf') / norm(q_exact, 'inf');
    relative_error_distance = [];

    % Gauss-Seidel iteration
    while relative_error(iterations) > tol && iterations <1000
        q_current = G * q_previous + C;

        % Relative error compared to the exact solution
        relative_error_distance(iterations) = norm(q_current - q_exact, 'inf') / norm(q_exact, 'inf');

        iterations = iterations + 1;
        q_previous = q_current;

        % Update relative error
        relative_error(iterations) = norm(q_exact - q_previous, 'inf') / norm(q_exact, 'inf');
    end

    % Final solution
    q_new = q_previous;

    % Plot relative error vs iterations (logarithmic scale)
    figure;
    semilogy(relative_error, '-o');
    hold on;
    semilogy(relative_error_distance, '--*');
    hold off;

    xlabel('Iterations');
    ylabel('Relative Error (log scale)');
    title('Relative Error vs Iterations (Logarithmic Scale)');
    legend('Error vs Previous', 'Error vs Exact Solution');
    grid on;
end

function [q_new, iterations] = jacobi(A, v, q_exact, tol, initial_guess)
    % Decompose matrix A into D (diagonal), L (lower), and U (upper) parts
    D = diag(diag(A));
    L = tril(A, -1);
    U = triu(A, 1);

    Q = L + D;
    C = D \ v;
    G = D \ (L+U);

    q_previous = initial_guess;
    iterations = 1;

    % Initialize error vectors
    relative_error = [];
    relative_error(iterations) = norm(q_exact - q_previous, 'inf') / norm(q_exact, 'inf');
    relative_error_distance = [];

    % Gauss-Seidel iteration
    while relative_error(iterations) > tol && iterations <1001
        q_current = G * q_previous + C;

        % Relative error compared to the exact solution
        relative_error_distance(iterations) = norm(q_current - q_exact, 'inf') / norm(q_exact, 'inf');

        iterations = iterations + 1;
        q_previous = q_current;

        % Update relative error
        relative_error(iterations) = norm(q_exact - q_previous, 'inf') / norm(q_exact, 'inf');
    end

    % Final solution
    q_new = q_previous;

    % Plot relative error vs iterations (logarithmic scale)
    figure;
    semilogy(relative_error, '-o');
    hold on;
    semilogy(relative_error_distance, '--*');
    hold off;

    xlabel('Iterations');
    ylabel('Relative Error (log scale)');
    title('Relative Error vs Iterations (Logarithmic Scale)');
    legend('Error vs Previous', 'Error vs Exact Solution');
    grid on;
end
function A = build_A(h, rho, M)
    % Build the matrix A according to the given 3D formula
    A = zeros(M, M);

    for m = 1:M
        for n = 1:M
            %Rmn = sqrt( ...
             %   (h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + ...
              %  (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2);
            Rmn =  ...
                (h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + ...
                (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2;

            A(m, n) = 1 / (4 * pi * Rmn);
        end
    end
end

function isDiagonallyDominant = checkDiagonalDominance(A)
    % Check if the matrix A is diagonally dominant
    n = size(A, 1);
    isDiagonallyDominant = true;

    for i = 1:n
        diag_element = abs(A(i,i));
        row_sum = sum(abs(A(i,:))) - diag_element;

        if diag_element <= row_sum
            isDiagonallyDominant = false;
            fprintf('Row %d is NOT diagonally dominant.\n', i);
            break;
        end
    end

    if isDiagonallyDominant
        fprintf('The matrix IS diagonally dominant.\n');
    end
end
