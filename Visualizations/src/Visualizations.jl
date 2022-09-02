module Visualizations
using Plots
using LinearAlgebra
using Statistics
using Utils

export determine_axes, get_quivers, blochplot, project

function unit(vec)
    vec./sqrt(dot(vec, vec))
end

function project(vectors, axis)
    vectors - unit(axis) * (unit(axis) * vectors)
end

function project(vectors, a1, a2)
    u1 = unit(a1)
    u2 = unit(a2)
    transpose(vectors) * u1, transpose(vectors) * u2
end

function findaxes(vectors)
    F = eigen(cov([vectors -vectors], dims=2))
    # vectors are given in order of increasing eigenvalue
    a = unit(F.vectors[:,1])
    b = unit(F.vectors[:,2])
    b, F.values[2], a, F.values[1]
end

function findrotation(v1, v2; mirror=true)
    if mirror
        v1 = [v1 -v1]
        v2 = [v2 -v2]
    end
    c1 = cov(v1, dims=2)
    c2 = cov(v2, dims=2)
    if det(c1) == 0 || det(c2) == 0
        return I
    end
    L = cov(v2, v1, dims=2) * c1
    F = svd(L)
    F.U * F.Vt
end

function determine_axes(sol)
    d, T, N = size(sol.u)
    rotations = []
    total_rotation = I(3)

    lastframe = sol.u[1:3,T,:]
    a1, s1, a2, s2= findaxes(lastframe)
    xaxes = [a1]
    yaxes = [a2]

    for i = T-1:-1:1
        frame1 = sol.u[1:3,i,:]
        frame2 = sol.u[1:3,i+1,:]
        U = findrotation(frame2, frame1)
        push!(rotations, U)
        x2, y2 = project(frame2, a1, a2)
        a1, a2 = U * a1, U * a2
        x1, y1 = project(frame1, a1, a2)
        
        Uc = findrotation(transpose([x2 y2]), transpose([x1 y1]); mirror=false)
        Tc = [a1 a2] * Uc
        a1 = Tc[:,1]
        a2 = Tc[:,2]
        
        push!(xaxes, a1)
        push!(yaxes, a2)
        total_rotation = U * total_rotation
    end
    
    rotations = rotations[end:-1:1];
    xaxes = xaxes[end:-1:1];
    yaxes = yaxes[end:-1:1];
    xaxes, yaxes, sqrt(s1)
end

function get_quivers(frame, xaxis, yaxis, scale)
    meanvec = unit(mean(frame, dims=2))[:,1]
    v1 = cross([0; 0; 1], meanvec).*scale #East
    v2 = cross(meanvec, v1) #North
    xv, yv = project([v1 v2], xaxis, yaxis)
    xv, yv
end

include("recipes.jl")

end
