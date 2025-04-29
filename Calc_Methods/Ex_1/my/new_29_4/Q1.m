%General Calculations and Values
% Parameters


% Define the vector q
q = [2;0;8;8;3;9;8;4;5;2;0;8;8;3;9;8;4;5];

% Calculate v = A * q


%---------------------------- Qestion 1-------------------------------

Q1 = figure('Visible', 'on');
movegui(Q1, 'west');

%---------------------------- Qestion 1a-------------------------------


% Solve using Gauss-Seidel method
M = 18;
rho = 1;
h = (rho * pi) / (M * 5); % As specified in the question
% Initialize the starting vector 


% Build the matrix A
[A , v] = build_A_v(h, rho, M, q, 'a');
StrCon = zeros(length(v), 1);
[q_1a , relative_error, relative_error_distance] = iterative_calculation(A, v, q, 10^(-3), StrCon, 'gauss_seidel');


%figure(Q1)
subplot(1,1,1); % 5 rows, 1 column, first plot
hold on;
semilogy(relative_error, '-o');
semilogy(relative_error_distance, '--*');
hold off;
xlabel('Iterations');
ylabel('Relative Error (log scale)');

title('Relative Error vs Iterations (Run 1a)');
grid on;

%---------------------------- Qestion 1b-------------------------------

%-------h = (rho * pi) / (M )--------
h = (rho * pi) / (M ); % As specified in the question
[A , v] = build_A_v(h, rho, M, q, 'b');
[q, relative_error, relative_error_distance] = iterative_calculation(A, v, q, 10^(-3), StrCon, 'gauss_seidel');
[relative_error_distance,relative_error,iter_plot] = Gauss_Seidel(A,v,q)

%add_subplot_Q1(relative_error, relative_error_distance, 2, Q1);
subplot(5,1,2); % 5 rows, 1 column, first plot
hold on;
semilogy(relative_error_1a, '-o');
semilogy(relative_error_distance, '--*');
hold off;
xlabel('Iterations');
ylabel('Relative Error (log scale)');

title('Relative Error vs Iterations (Run 1b1)');
grid on;

%-------h = (rho * pi) / (M * 2 )--------
h = (rho * pi) / (M * 2); % As specified in the question
[A , v] = build_A_v(h, rho, M, q, 'b');
[q_1b2, relative_error, relative_error_distance] = iterative_calculation(A, v, q, 10^(-3), StrCon, 'gauss_seidel');
add_subplot_Q1(relative_error, relative_error_distance, 3, Q1);

subplot(5,1,3); % 5 rows, 1 column, first plot
hold on;
semilogy(relative_error_1a, '-o');
semilogy(relative_error_distance, '--*');
hold off;
xlabel('Iterations');
ylabel('Relative Error (log scale)');

title('Relative Error vs Iterations (Run 1b2)');
grid on;
%---------------------------- Qestion 1c-------------------------------

h = (rho * pi) / (M * 5); % As specified in the questio
[A , v] = build_A_v(h, rho, M, q, 'c');

[q_1c, irelative_error, relative_error_distance] = iterative_calculation(A, v, q, 10^(-3), StrCon, 'jacobi');
%add_subplot_Q1(relative_error, relative_error_distance, 4, Q1);
subplot(5,1,4); % 5 rows, 1 column, first plot
hold on;
semilogy(relative_error_1a, '-o');
semilogy(relative_error_distance, '--*');
hold off;
xlabel('Iterations');
ylabel('Relative Error (log scale)');

title('Relative Error vs Iterations (Run 1c)');
grid on;

%---------------------------- Qestion 1d-------------------------------
h = (rho * pi) / (M * 5); % As specified in the questio
[A , v] = build_A_v(h, rho, M, q, 'd');

[q_1d, relative_error, relative_error_distance] = iterative_calculation(A, v, q, 10^(-3), StrCon, 'jacobi');
%add_subplot_Q1(relative_error, relative_error_distance, 5, Q1);
subplot(5,1,5); % 5 rows, 1 column, first plot
hold on;
semilogy(relative_error_1a, '-o');
semilogy(relative_error_distance, '--*');
hold off;
xlabel('Iterations');
ylabel('Relative Error (log scale)');

title('Relative Error vs Iterations (Run 1d)');
grid on;





%---------------------------- Qestion 2-------------------------------

Q2_fig = figure;
movegui(Q2_fig, 'east');
%---------------------------- Qestion 2a-------------------------------

h = (rho * pi *10) / (M); % As specified in the questioמ
[A , v] = build_A_v(h, rho, M, q, 'a');
detA = det(A)

%---------------------------- Qestion 2b-------------------------------
h = (rho * pi *10) / (M); % As specified in the questio
[A , v] = build_A_v(h, rho, M, q, 'b');
for i = 1:0.5:5
    disp(i)
end

 




%% --- Functions ---


%% ------------------------------jacobi & gauss_seidel------------------------------------------------------------------------------
function [Rel_dist,Rel_err_real,iter_plot] = Gauss_Seidel(A,v,q)
     L = tril(A,-1);
     D = diag(diag(A));
     Q = L + D ;
     neg_u = Q - A; %-U
     inv_Q = inv(Q);
     G = inv_Q * neg_u ; % -u * (L+D)^(-1)
     C = inv_Q * v; % (L+D)^(-1) * v
     Err_Endurance = 10 ^ (-3);
     q_k = C; %for q^(1), k=1
     Rel_dist = zeros;
     Rel_err_real = zeros;
     iter = 1;
     iter_plot = zeros;
     err = max(abs(q-q_k));
     max_iter = 500; %Limit Iterations
     while abs(err) > Err_Endurance && iter <=max_iter
         q_k_minus_1 = q_k;
         q_k = G*(q_k_minus_1) + C; % q^(k)=-u*(L+D)^(-1) * q^(k-1) +(L+D)^(-1)*v
         err = norm(q-q_k,'inf');
         Rel_dist(iter) = norm(q_k - q_k_minus_1, 'inf') /norm(q_k_minus_1, 'inf');
         Rel_err_real(iter) = norm(q_k - q,'inf') / norm(q,'inf');
         iter_plot(iter) = iter;
         iter = iter + 1;
     end
end
function [q_new, relative_error, relative_error_distance] = iterative_calculation(A, v, q_exact, tol, StrCon, type)
    % Decompose matrix A into D (diagonal), L (lower), and U (upper) parts
    D = diag(diag(A));
    L = tril(A, -1);
    Q = L + D;
    U =  Q - A;
    

    if strcmp(type, 'jacobi')
      
        C = D \ v;
        G = D \ (L+U);

    elseif strcmp(type, 'gauss_seidel')
        
        C = Q \ v;
        G = Q \ (-U);
    

    end

    q_previous = StrCon;
    iterations = 1;
    q_current = C % iteration 0

    % Initialize error vectors
    relative_error = [];
    relative_error_distance = [];
    
    err = abs(max(q_exact-q_current));

    % Start iteration
    while err > tol && iterations <1001
        
        q_previous = q_current;
        q_current = G * q_previous + C;

        % Relative error compared to the exact solution
        relative_error_distance(iterations) = norm(q_current - q_previous, 'inf') / norm(q_previous, 'inf');
        relative_error(iterations) = norm(q_current - q_exact, 'inf') / norm(q_exact, 'inf');
        iterations = iterations + 1;
        err = abs(norm(q_exact-q_current,'inf'));

        % Update relative error
        
    end

    % Final solution
    q_new = q_previous;

    %add_subplot_Q1(relative_error, relative_error_distance, 1, Q1, section);

    

    

end
%% ------------------------------Plot Figure------------------------------------------------------------------------------
function add_subplot_Q1(relative_error_1a, relative_error_distance, subplot_index, Q1, section)
    figure(Q1)
    subplot(5,1,subplot_index); % 5 rows, 1 column, first plot
    hold on;
    semilogy(relative_error_1a, '-o');
    semilogy(relative_error_distance, '--*');
    hold off;
    xlabel('Iterations');
    ylabel('Relative Error (log scale)');
    %if strcmp(mat_calc, 'd')
    title('Relative Error vs Iterations (Run 1a)');
    legend('Error vs Previous (1a)', 'Error vs Exact (1a)');
    grid on;
end
%% ------------------------------Build A------------------------------------------------------------------------------

function [A , v]= build_A_v(h, rho, M, q ,mat_calc)
    % Build the matrix A according to the given 3D formula
    A = zeros(M, M);

    for m = 1:M
        for n = 1:M
            if strcmp(mat_calc, 'd')
                Rmn = ((h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2);
            else
                Rmn = sqrt((h + rho * sin(m * pi / M) - rho * sin(n * pi / M))^2 + (rho * cos(m * pi / M) - rho * cos(n * pi / M))^2);
            end
            A(m, n) = 1 / (4 * pi * Rmn);
        end
    end
    v = A * q;
end
%% ------------------------------Extras--------------------------------------------------------------

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
