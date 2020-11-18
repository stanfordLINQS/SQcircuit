using LinearAlgebra
using PyPlot
using DelimitedFiles

#const CP =

# Operator for φ in the φ basis


const ħ=1.0545718e-34;
const Φ0 = 2.067833848e-15;
const e0 = 1.60217662e-19;
const kB = 1.38064852e-23;




C1 = 15e-15;
C2 = 10e-15;
L12 = 50e-9;
EJ = ħ*2*pi*20e9;
Q2 = 0;

# _v2 is the slightly more generic scheme, see notes
function get_circ1_params_v2(C1, C2, L12, EJ, Q2)

    #c1 = -1
    #c2 = 1
    c1=0
    c2=1


    C1t = 1/C1 + 1/C2;
    C1t = 1/C1t;

    C2t = C1

    C12t = C1;

    L1t = L12;

    ω1 = 1/sqrt(L1t*C1t);
    qzp = sqrt(ħ*ω1*C1t/2);
    ϕzp = sqrt(ħ*ω1*L1t/2);

    EC = (2*e0*c2)^2/(2*C2t);
    Eκ = (2*e0*c2)*qzp/(C12t);

    expfactor = 1im*ϕzp*2π/Φ0*c1;


    return [ω1, qzp, ϕzp, C2t, C12t, EC, Eκ, expfactor, Q2/(2*e0*c2)]
end

# _v2 is the slightly more generic scheme, see notes
function get_circ2_params_v2(C1, C2, L12, EJ, Q2)

    c1 = 1/sqrt(2)
    c2 = 1/sqrt(2)
    #c1 = sqrt(2)
    #c2 = 0

    C1t = 1/(2*C1) + 1/(2*C2);
    C1t = 1/C1t;

    C2t = 1/(2*C1) + 1/(2*C2);
    C2t = 1/C2t;

    C12t = 1/(2*C2) - 1/(2*C1)
    C12t = 1/C12t;

    L1t = L12/2;


    ω1 = 1/sqrt(L1t*C1t);
    qzp = sqrt(ħ*ω1*C1t/2);
    ϕzp = sqrt(ħ*ω1*L1t/2);

    EC = (2*e0*c2)^2/(2*C2t);
    Eκ = (2*e0*c2)*qzp/(C12t);

    expfactor = 1im*ϕzp*2π/Φ0*c1;

    return [ω1, qzp, ϕzp, C2t, C12t, EC, Eκ, expfactor, Q2/(2*e0*c2)]
end





get_circ1_params_v2(C1,C2,L12,EJ,Q2)
get_circ2_params_v2(C1,C2,L12,EJ,Q2)



## _v3 is even more generic, does rotation. see TeX notes

function get_rotated_matrices(R,C1,C2,L12)


    Cmat = [C1 0; 0 C2];
    Lstar = [1/L12 -1/L12; -1/L12 1/L12]

    S = (R')^(-1);
    Rinv = R^(-1);
    Sinv = R';

    Lstar_t = Rinv'*Lstar*Rinv;
    # println("rotated Lstar matrix: ")
    # println(Lstar_t)

    Cmatinv = Cmat^(-1);

    Cmatinv_t = Sinv'*Cmatinv*Sinv;
    # println("****************************************************************************************************************************")
    # println(Sinv)
    # println(Cmatinv)
    # println(Cmatinv_t)


    Evec = [1 0];

    C1t = 1/Cmatinv_t[1,1];
    C2t = 1/Cmatinv_t[2,2];
    C12t = -1/Cmatinv_t[1,2];

    L1t = 1/Lstar_t[1,1]


    c1 = (Evec*Rinv)[1]
    c2 = (Evec*Rinv)[2]


    return [C1t, C2t, C12t, L1t, c1, c2]

end

function get_generic_circ_params(R,C1, C2, L12, EJ, Q2)


    (C1t, C2t, C12t, L1t, c1, c2) = get_rotated_matrices(R,C1,C2,L12)


    ω1 = 1/sqrt(L1t*C1t);
    qzp = sqrt(ħ*ω1*C1t/2);
    ϕzp = sqrt(ħ*ω1*L1t/2);

    EC = (2*e0*c2)^2/(2*C2t);
    Eκ = (2*e0*c2)*qzp/(C12t);

    expfactor = 1im*ϕzp*2π/Φ0*c1;


    return [ω1, qzp, ϕzp, C2t, C12t, EC, Eκ, expfactor, Q2/(2*e0*c2)]
end

function get_circ1_params_v3(C1, C2, L12, EJ, Q2)
    R = [0 1; 1 1]^(-1);
    #R = [1 -1; 1 1]
    return get_generic_circ_params(R,C1, C2, L12, EJ, Q2)
end
function get_circ2_params_v3(C1, C2, L12, EJ, Q2)
    R = [1 -1; 1 1]/sqrt(2)
    return get_generic_circ_params(R,C1, C2, L12, EJ, Q2)
end

get_circ1_params_v2(C1,C2,L12,EJ,Q2)-get_circ1_params_v3(C1,C2,L12,EJ,Q2)
get_circ2_params_v2(C1,C2,L12,EJ,Q2)-get_circ2_params_v3(C1,C2,L12,EJ,Q2)

function get_n_op(solver_params)

    N = solver_params["N"]
    M = solver_params["M"]
    n_op =  (Complex{Float64})[(i==j)*(i-M/2) for i in 1:M, j in 1:M]
    id = (Complex{Float64})[(i==j)*1 for i in 1:N, j in 1:N]

    return kron(id,n_op)

end

function get_nn1_op(solver_params) # sum_n |n><n-1|
    N = solver_params["N"]
    M = solver_params["M"]
    nn1 =  (Complex{Float64})[(i==(j+1)) for i in 1:M, j in 1:M]
    id = (Complex{Float64})[(i==j)*1 for i in 1:N, j in 1:N]
    return kron(id,nn1)
end

function get_id(solver_params)
    N = solver_params["N"]
    M = solver_params["M"]
    return  (Complex{Float64})[(i==j)*1 for i in 1:N*M, j in 1:N*M]
end


# annihilation operator in the photon number basis
function get_a_op(solver_params)

    N = solver_params["N"]
    M = solver_params["M"]
    a =  (Complex{Float64})[(i==j-1)*sqrt(i) for i in 1:N, j in 1:N]
    id = (Complex{Float64})[(i==j)*1 for i in 1:M, j in 1:M]
    return kron(a,id)
end


function get_D_alpha(solver_params,alpha)
    N = solver_params["N"]
    M = solver_params["M"]
    a =  (Complex{Float64})[(i==j-1)*sqrt(i) for i in 1:N, j in 1:N]
    id = (Complex{Float64})[(i==j)*1 for i in 1:M, j in 1:M]

    Dα = exp(alpha*a'-alpha'*a)

    return kron(Dα,id)
end


function get_H(circ_param_method,C1,C2,L12,EJ,Q2)

    #[ω1, qzp, ϕzp, C2t, C12t, EC, Eκ, expfactor, Q2] =
    (ω1, qzp, ϕzp, C2t, C12t, EC, Eκ, expfactor, Q2)=circ_param_method(C1,C2,L12,EJ,Q2)
    N = solver_params["N"]
    M = solver_params["M"]
    dim = N*M;

    a = get_a_op(solver_params);
    n = get_n_op(solver_params);
    Dα = get_D_alpha(solver_params,expfactor)
    nn1 = get_nn1_op(solver_params)
    id = get_id(solver_params)

    H = zeros((Complex{Float64}),dim,dim);

    H = (ħ*ω1*a'a + EC*(id.*Q2+n)^2 + 1.0im*Eκ*(a-a')*(id.*Q2+n) - EJ/2 *(Dα*nn1 + Dα'*nn1'))/(ħ*ω1)
    HJJ =  EJ/2 *(Dα*nn1 + Dα'*nn1')/ħ
    open("HJJ.txt", "w") do io
        writedlm(io, HJJ)
    end

    # for i in 1:9
    #     for j in 1:9
    #         print(XX[i,j],",")
    #     end
    #     println()
    # end

end

solver_params = Dict()
solver_params["N"]= 10  # fock basis
solver_params["M"]= 10  # charge basis dimension

nlvl = 10;

println("====================================================================================================================================")

H1 = get_H(get_circ1_params_v3,C1,C2,L12,EJ,0)
# F1 = eigen(H1)
# ΔE1s= F1.values[1:nlvl]
# println(ΔE1s)

H2 = get_H(get_circ2_params_v3,C1,C2,L12,EJ,0)
# F2 = eigen(H2)
# ΔE2s= F2.values[1:nlvl]
# println(ΔE2s)


#
# Q2s = (-0.5:0.1:0.5)*(2*e0)
#
#
# ΔE1s = zeros(length(Q2s),nlvl)
# ΔE2s = zeros(length(Q2s),nlvl)
#
# for k in range(1,stop=length(Q2s))
#     println(Q2s[k])
#     H1 = get_H(get_circ1_params_v3,C1,C2,L12,EJ,Q2s[k])
#     F1 = eigen(H1)
#     ΔE1s[k,:]= F1.values[1:nlvl]#F1.values[2:(nlvl+1)]-F1.values[1:nlvl]
#
#     H2 = get_H(get_circ2_params_v3,C1,C2,L12,EJ,Q2s[k]/sqrt(2))
#     F2 = eigen(H2)
#     ΔE2s[k,:]= F2.values[1:nlvl]#F2.values[2:(nlvl+1)]-F2.values[1:nlvl]
# end
#
#
# figure()
# subplot(121)
# plot(Q2s/2/e0.+0.6,ΔE1s)
# plot(Q2s/2/e0.-0.6,ΔE2s)
# plot([0,0],ylim())
# ylabel("Energies")
# subplot(122)
#
# plot(Q2s/2/e0,ΔE1s-ΔE2s)
# ylabel("Energy diff between two solutions")
# gcf()
#
#
#
# H2 = get_H(get_circ2_params_v3,C1,C2,L12,EJ,-0*2*e0)
# F2 = eigen(H2)
#
# H1 = get_H(get_circ1_params_v3,C1,C2,L12,EJ,-0.5*2*e0)
# F1 = eigen(H1)
#
#
# println("ΔE for the first levels")
# println(F1.values[2:(nlvl+1)]-F1.values[1:nlvl])
# println(F2.values[2:(nlvl+1)]-F2.values[1:nlvl])
#
# ##
# figure()
# plot(abs.(F2.vectors[:,3]).^2)
# gcf()
# close(gcf())
#
# #
#
# figure()
# subplot(121)
# plot(Q2s/2/e0,ΔE1s)
# subplot(122)
# plot(Q2s/2/e0,ΔE2s)
# gcf()
