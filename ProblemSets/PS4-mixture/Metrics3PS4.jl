using Distributions, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

function ps4()
#Question 1

url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2021/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occ_code

#1.

function mlogit_with_Z(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    J = length(unique(y))
    K = size(X,2)
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==J
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T,N,J)
    dem = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)
    loglike = -sum( bigY.*log.(P))
    return loglike
end

startvals=[0.05570767876416688, 0.08342649976722213, -2.344887681361976, 0.04500076157943125, 0.7365771540890512, -3.153244238810631, 0.09264606406280998, -0.08417701777996893, -4.273280002738097, 0.023903455659102114, 0.7230648923377259, -3.749393470343111, 0.03608733246865346, -0.6437658344513095, -4.2796847340030375, 0.0853109465190059, -1.1714299392376775, -6.678677013966667, 0.086620198654063, -0.7978777029320784, -4.969132023685069, -0.0941942241795243];

td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward);

theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))


theta_hat_mle_ad = theta_hat_optim_ad.minimizer
println(theta_hat_mle_ad)
H = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))

println([theta_hat_mle_ad theta_hat_mle_ad_se])

#Question 2
#The previous coefficient did not make a lot of sense because it was negative;however, this coefficient is positive. So it makes more sense.
#Wage and utility should be positively related, which was why the last coefficient's negative value was troubling.

#Question 3

#3A For Practice
using Distributions
cd("C:/Users/wat20/OneDrive/fall-2021-master (1).zip/fall-2021-master/ProblemSets/PS4-mixture")

include("lgwt.jl") # make sure the function gets read in

# define distribution
d = Normal(0,1) # mean=0, standard deviation=1

# get quadrature nodes and weights for 7 grid points
nodes, weights = lgwt(7,-4,4)
# now compute the integral over the density and verify its 1
sum(weights.*pdf.(d,nodes))
# now compute the expectation and verify it's 0
sum(weights.*nodes.*pdf.(d,nodes))

#3B
#Because sigma is 2 we double the 5σ value to be from -10 to 10 instead of -5 to 5
nodes, weights = lgwt(7,-10,10) 
d = Normal(0,2)
sum(weights.*pdf.(d,nodes))
println(sum((nodes.^2).*weights.*pdf.(d,nodes)))

nodes, weights = lgwt(10,-10,10)
d = Normal(0,2)
sum(weights.*pdf.(d,nodes))
println(sum((nodes.^2).*weights.*pdf.(d,nodes)))
#Yes, I believe these estimates work well.

#3C
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad] 
N = length(y)
K = size(X, 2)

rng = MersenneTwister(8675309)
rdraw = (rand!(rng, zeros(1000000)).-(.5)).*20
d = Normal(0,2)
20*mean(rdraw.^2 .*pdf.(d,rdraw))
#The value approaches 4
20*mean(rdraw .*pdf.(d,rdraw))
# Close to 0 

20*mean(pdf.(d,rdraw))
#This approaches 1

rdraw = (rand!(rng, zeros(1000)).-(.5)).*20
d = Normal(0,2)
20*mean(rdraw.^2 .*pdf.(d,rdraw))
#This is close to 1
20*mean(rdraw .*pdf.(d,rdraw))
#A little far from 0 
20*mean(pdf.(d,rdraw))
#Close to 1

#3D

#Question 4
#I'm very unsure of my approach, I'm sure I'm missing something here.
#I had to eventually just stop on 4 and 5, although I did try to think hard on these.
#My idea was to run a different function but keep the same interior components?
function mlogit_with_Z_quad(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    J = length(unique(y))
    K = size(X,2)
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==J
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T,N,J)
    dem = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)
    loglike = -sum( bigY.*log.(P))
    return loglike
end


#Question 5
#I tried to go about a similar structure as problem 4, but I'm pretty sure I'm still missing something.
function mlogit_with_Z_mcmc(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]
    J = length(unique(y))
    K = size(X,2)
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==J
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T,N,J)
    dem = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end

    P = num./repeat(dem,1,J)
    loglike = -sum( bigY.*log.(P))
    return loglike
end


ps4()