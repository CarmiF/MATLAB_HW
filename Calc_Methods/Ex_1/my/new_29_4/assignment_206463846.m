%---------------------------- General-------------------------------


%---------------------------- Qestion 1-------------------------------
M = 18;
rho = 1;
h = (rho * pi) / (M * 5);
A = build_A(h, rho, M ,'sqrt');
q_exact = [2;0;8;8;3;9;8;4;5;2;0;8;8;3;9;8;4;5];
tol = 10^-3;
v = A * q_exact;
StrCon = zeros(length(v), 1);


[q_gauss_seidel, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon);

%---------------------------- Qestion 1a-------------------------------


%---------------------------- Qestion 1b-------------------------------

%-------h = (rho * pi) / (M )------------


%-------h = (rho * pi) / (M * 2 )--------

%---------------------------- Qestion 1c-------------------------------

%---------------------------- Qestion 1d-------------------------------


%---------------------------- Qestion 2-------------------------------


%---------------------------- Qestion 2a-------------------------------


%---------------------------- Qestion 2b-------------------------------



%% --- Functions ---


%% ------------------------------jacobi & gauss_seidel------------------------------------------------------------------------------
function [q_current, real_err, rel_dis] = gauss_seidel(A, v, q_exact, tol, StrCon)
    D = diag(diag(A));
    L = tril(A, -1);
    Q = L + D;
    U =  Q - A;
    C = Q \ v;
    G = Q \ (U);
    q_previous = C;
    i = 1;
    err = max(abs((q_exact-q_previous)./q_exact));
    rel_dis = [];
    while err > tol && i <1001
        
        real_err(i) = err;
        q_current = G * q_previous + C;
        rel_dis(i) = max(abs((q_current-q_previous)./q_previous));
        err = max(abs((q_exact-q_current)./q_exact));
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


